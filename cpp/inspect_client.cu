// inspect_client.cu - Phase 2: the three-stage inspection line, under live load.
//
// Built on the paced client of sadbodhs/computer_vision_optimization
// (cpp/src/grpc_client_cuda.cu, flows B2/B3): the letterbox and compact kernels,
// CUDA shared memory with Triton, and open-loop virtual cameras whose frames
// are due on a fixed schedule whether or not the server has kept up.
//
// Per camera, per frame:
//   stage 1  upload the preprocessed 640 frame into a CUDA shm region, run the
//            locator (yolov8s), compact its candidates on the GPU and NMS on the
//            CPU, exactly as flow B2 does
//   crops    K seeded boxes per frame (25-50% of the frame side, seeded per
//            camera/frame/crop so every config sees the same crops), cut from the
//            frame ON THE GPU and bilinearly resized to S x S into the stage-2
//            input region. Fixed crops, not detections: the parts in the replay
//            video are not the parts on a line, and K must be a controlled setting.
//   stage 2  ONE request of batch K to the anomaly model
//   stage 3  each crop is flagged with seeded probability p; a flagged crop (or
//            the whole frame, resized to the stage-3 size) goes to the stage-3 model
//            ASYNCHRONOUSLY, from a pool of slots. If every slot is busy the crop
//            is counted as skipped: a slow stage 3 must not stall the line, and
//            skips are reported, never hidden.
//
// Timing, all from the frame's DUE time (no coordinated omission):
//   fast path   due -> stage-2 result    (what keeps the line moving)
//   explain     due -> stage-3 result    (how long an operator waits for a name)
//
// Output: one JSON line.
#include <cuda_runtime_api.h>
#define TRITON_ENABLE_GPU 1
#include <grpc_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace tc = triton::client;
using clk = std::chrono::steady_clock;

#define CUDA_CHECK(x)                                                          \
  do { cudaError_t e = (x); if (e != cudaSuccess) {                             \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " @" << __LINE__ << std::endl; exit(1); } \
  } while (0)
#define CHECK_OK(x)                                                            \
  do { tc::Error err = (x); if (!err.IsOk()) {                                  \
    std::cerr << "client error: " << err << " @" << __LINE__ << std::endl; exit(1); } } while (0)

static const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400, MAX_DETS = 4096;

// ---------------- kernels ----------------
struct GPUDet { float x1, y1, x2, y2, score; int cls; };

// Same kernel as flow B2 (grpc_client_cuda.cu).
__global__ void compact_candidates_kernel(
    const float* __restrict__ out, int num_classes, int num_anchors, float conf_thr,
    GPUDet* __restrict__ dets, int* __restrict__ d_count, int max_dets) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  float best = 0.f; int best_c = -1;
  for (int c = 0; c < num_classes; ++c) {
    float s = out[(4 + c) * num_anchors + a];
    if (s > best) { best = s; best_c = c; }
  }
  if (best < conf_thr) return;
  float cx = out[0 * num_anchors + a], cy = out[1 * num_anchors + a];
  float w = out[2 * num_anchors + a], h = out[3 * num_anchors + a];
  int slot = atomicAdd(d_count, 1);
  if (slot < max_dets) dets[slot] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, best_c};
}

// Crop a box from a CHW float frame and bilinearly resize it to S x S, into
// dst (CHW). Center-aligned sampling, as cv2 INTER_LINEAR and the A2/B2 kernel.
// Used for both stage-2 crops and stage-3 inputs (crop, or the whole frame).
__global__ void crop_resize_kernel(const float* __restrict__ src, int H, int W,
                                   float bx, float by, float bw, float bh,
                                   float* __restrict__ dst, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float fx = bx + ((float)x + 0.5f) * bw / (float)S - 0.5f;
  float fy = by + ((float)y + 0.5f) * bh / (float)S - 0.5f;
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float ax = fx - x0, ay = fy - y0;
  int x0c = min(max(x0, 0), W - 1), x1c = min(max(x0 + 1, 0), W - 1);
  int y0c = min(max(y0, 0), H - 1), y1c = min(max(y0 + 1, 0), H - 1);
  size_t plane = (size_t)H * W, oplane = (size_t)S * S;
  for (int c = 0; c < 3; ++c) {
    const float* p = src + c * plane;
    float v = (p[(size_t)y0c * W + x0c] * (1 - ax) + p[(size_t)y0c * W + x1c] * ax) * (1 - ay)
            + (p[(size_t)y1c * W + x0c] * (1 - ax) + p[(size_t)y1c * W + x1c] * ax) * ay;
    dst[c * oplane + (size_t)y * S + x] = v;
  }
}

static int nms_count(const GPUDet* dets, int n, float iou_thr) {
  std::vector<GPUDet> v(dets, dets + n);
  std::sort(v.begin(), v.end(), [](const GPUDet& a, const GPUDet& b) { return a.score > b.score; });
  std::vector<bool> removed(v.size(), false);
  int kept = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (removed[i]) continue;
    kept++;
    for (size_t j = i + 1; j < v.size(); ++j) {
      if (removed[j] || v[i].cls != v[j].cls) continue;
      float ix1 = std::max(v[i].x1, v[j].x1), iy1 = std::max(v[i].y1, v[j].y1);
      float ix2 = std::min(v[i].x2, v[j].x2), iy2 = std::min(v[i].y2, v[j].y2);
      float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
      float uni = (v[i].x2 - v[i].x1) * (v[i].y2 - v[i].y1) +
                  (v[j].x2 - v[j].x1) * (v[j].y2 - v[j].y1) - inter;
      if (inter / std::max(uni, 1e-6f) > iou_thr) removed[j] = true;
    }
  }
  return kept;
}

// Deterministic per-(seed, camera, frame, crop, salt) uniform in [0,1): every
// config sees the same crops and the same flags.
static double u01(unsigned seed, int cam, long frame, int crop, int salt) {
  uint64_t z = (uint64_t)seed * 0x9E3779B97F4A7C15ull ^ (uint64_t)cam * 0xBF58476D1CE4E5B9ull
             ^ (uint64_t)frame * 0x94D049BB133111EBull ^ (uint64_t)crop * 0xD6E8FEB86659FD93ull
             ^ (uint64_t)salt * 0x2545F4914F6CDD1Dull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------- model I/O from Triton's metadata ----------------
struct IOSpec { std::string name, dtype; std::vector<int64_t> dims; bool batched; size_t per_sample_bytes; };

static size_t dtype_bytes(const std::string& t) {
  if (t == "FP32" || t == "INT32") return 4;
  if (t == "FP16" || t == "BF16") return 2;
  if (t == "INT64" || t == "FP64") return 8;
  return 1;
}

static void model_io(tc::InferenceServerGrpcClient* c, const std::string& model,
                     IOSpec& in, std::vector<IOSpec>& outs) {
  inference::ModelMetadataResponse md;
  CHECK_OK(c->ModelMetadata(&md, model));
  auto conv = [](const inference::ModelMetadataResponse::TensorMetadata& t) {
    IOSpec s{t.name(), t.datatype(), {}, false, 0};
    size_t n = 1;
    for (int i = 0; i < t.shape_size(); ++i) {
      int64_t d = t.shape(i);
      s.dims.push_back(d);
      if (i == 0 && d == -1) { s.batched = true; continue; }
      n *= (size_t)std::max<int64_t>(d, 1);
    }
    s.per_sample_bytes = n * dtype_bytes(s.dtype);
    return s;
  };
  in = conv(md.inputs(0));
  for (int i = 0; i < md.outputs_size(); ++i) outs.push_back(conv(md.outputs(i)));
}

// ---------------- shared state ----------------
static std::vector<char> g_frame_buf;   // frames.bin, loaded once, read-only

struct Opts {
  std::string url = "localhost:8101", stage1 = "yolov8s", stage2 = "", stage3 = "none";
  std::string bls = "";   // Phase 4: one server-side request per frame (Triton BLS model)
  std::string file = "frames.bin", s3_input = "crop", phase = "random";
  int streams = 1, k = 4, s2_size = 256, s3_size = 384, s3_slots = 4;
  double fps = 30, duration = 15, warmup = 1, p = 0;
  double s3_budget = 0, s3_cost_ms = 0;   // Phase 3: stage-3 share of GPU time (0 = no budget)
  unsigned seed = 1;
};

struct Stats {
  std::mutex mtx;
  std::vector<double> fast, explain;
  double t_s1 = 0, t_crop = 0, t_s2 = 0; long n = 0;
  std::atomic<long> frames{0}, late{0}, flagged{0}, s3_sent{0}, s3_done{0}, s3_skipped{0}, s3_shed{0}, dets{0};
};

// Phase 3: admission control for stage 3 by GPU time. One token bucket shared
// by every camera, filled with milliseconds of GPU time at s3_budget x 1000 ms
// per second, capped at 100 ms of wall time's worth (at least one request).
// A flagged crop must pay its measured stage-3 cost (Phase 1c) before it is
// sent; if it cannot, it is SHED and counted. Unlike the slot pool, this bounds
// stage 3's share of the GPU whatever the fast path is doing.
struct Budget {
  std::mutex m;
  bool on = false;
  double rate_ms_per_s = 0, cap_ms = 0, tokens_ms = 0;
  clk::time_point last = clk::now();
  bool take(double cost) {
    if (!on) return true;
    std::lock_guard<std::mutex> lk(m);
    auto now = clk::now();
    tokens_ms = std::min(cap_ms, tokens_ms + rate_ms_per_s * std::chrono::duration<double>(now - last).count());
    last = now;
    if (tokens_ms < cost) return false;
    tokens_ms -= cost;
    return true;
  }
};
static Budget g_budget;

struct S3Slot {
  std::atomic<bool> busy{false};
  clk::time_point due; bool record = false;
  tc::InferInput* in = nullptr;
  std::vector<tc::InferRequestedOutput*> outs;
};

static void run_camera(const Opts o, int cam, clk::time_point t_start, Stats* st) {
  const size_t S1_IN = (size_t)3 * IMG * IMG * sizeof(float);
  const size_t S1_OUT = (size_t)84 * NUM_ANCHORS * sizeof(float);
  const size_t S2_IN_ONE = (size_t)3 * o.s2_size * o.s2_size * sizeof(float);
  const size_t S3_IN_ONE = (size_t)3 * o.s3_size * o.s3_size * sizeof(float);
  const bool use_s3 = o.stage3 != "none";
  const std::string tag = std::to_string(cam) + "_" + std::to_string(getpid());

  std::unique_ptr<tc::InferenceServerGrpcClient> c, c3;
  CHECK_OK(tc::InferenceServerGrpcClient::Create(&c, o.url, false));
  if (use_s3) CHECK_OK(tc::InferenceServerGrpcClient::Create(&c3, o.url, false));

  // --- stage 1 regions (as B2) ---
  float *s1_in, *s1_out;
  CUDA_CHECK(cudaMalloc(&s1_in, S1_IN));
  CUDA_CHECK(cudaMalloc(&s1_out, S1_OUT));
  cudaIpcMemHandle_t h;
  CUDA_CHECK(cudaIpcGetMemHandle(&h, s1_in));  CHECK_OK(c->RegisterCudaSharedMemory("s1i_" + tag, h, 0, S1_IN));
  CUDA_CHECK(cudaIpcGetMemHandle(&h, s1_out)); CHECK_OK(c->RegisterCudaSharedMemory("s1o_" + tag, h, 0, S1_OUT));
  tc::InferInput* s1i; CHECK_OK(tc::InferInput::Create(&s1i, "images", {1, 3, IMG, IMG}, "FP32"));
  CHECK_OK(s1i->SetSharedMemory("s1i_" + tag, S1_IN, 0));
  tc::InferRequestedOutput* s1o; CHECK_OK(tc::InferRequestedOutput::Create(&s1o, "output0"));
  CHECK_OK(s1o->SetSharedMemory("s1o_" + tag, S1_OUT, 0));
  tc::InferOptions opt1(o.stage1);

  // --- Phase 4 (BLS): one request per frame carries the frame (same shm region)
  // and the K crop boxes; the server runs stage 1, the post, the crops and stage 2.
  tc::InferInput* bls_img = nullptr; tc::InferInput* bls_boxes = nullptr;
  tc::InferRequestedOutput* bls_score = nullptr;
  if (!o.bls.empty()) {
    CHECK_OK(tc::InferInput::Create(&bls_img, "images", {1, 3, IMG, IMG}, "FP32"));
    CHECK_OK(bls_img->SetSharedMemory("s1i_" + tag, S1_IN, 0));
    CHECK_OK(tc::InferInput::Create(&bls_boxes, "boxes", {o.k, 4}, "FP32"));
    CHECK_OK(tc::InferRequestedOutput::Create(&bls_score, "score"));
  }
  tc::InferOptions optb(o.bls.empty() ? o.stage1 : o.bls);
  std::vector<float> box_buf(4 * o.k);

  // --- stage 2 regions: one input of batch K, outputs from metadata ---
  IOSpec in2; std::vector<IOSpec> outs2;
  model_io(c.get(), o.stage2, in2, outs2);
  float* s2_in; CUDA_CHECK(cudaMalloc(&s2_in, S2_IN_ONE * o.k));
  CUDA_CHECK(cudaIpcGetMemHandle(&h, s2_in)); CHECK_OK(c->RegisterCudaSharedMemory("s2i_" + tag, h, 0, S2_IN_ONE * o.k));
  tc::InferInput* s2i;
  CHECK_OK(tc::InferInput::Create(&s2i, in2.name, {o.k, 3, o.s2_size, o.s2_size}, "FP32"));
  CHECK_OK(s2i->SetSharedMemory("s2i_" + tag, S2_IN_ONE * o.k, 0));
  std::vector<const tc::InferRequestedOutput*> s2o;
  for (size_t i = 0; i < outs2.size(); ++i) {
    size_t b = outs2[i].per_sample_bytes * o.k;
    void* p; CUDA_CHECK(cudaMalloc(&p, b));
    std::string r = "s2o" + std::to_string(i) + "_" + tag;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, p)); CHECK_OK(c->RegisterCudaSharedMemory(r, h, 0, b));
    tc::InferRequestedOutput* ro; CHECK_OK(tc::InferRequestedOutput::Create(&ro, outs2[i].name));
    CHECK_OK(ro->SetSharedMemory(r, b, 0));
    s2o.push_back(ro);
  }
  tc::InferOptions opt2(o.stage2);

  // --- stage 3: a pool of slots, each with its own input/output offsets ---
  std::vector<std::unique_ptr<S3Slot>> slots;
  float* s3_in = nullptr;
  if (use_s3) {
    IOSpec in3; std::vector<IOSpec> outs3;
    model_io(c3.get(), o.stage3, in3, outs3);
    CUDA_CHECK(cudaMalloc(&s3_in, S3_IN_ONE * o.s3_slots));
    CUDA_CHECK(cudaIpcGetMemHandle(&h, s3_in));
    CHECK_OK(c3->RegisterCudaSharedMemory("s3i_" + tag, h, 0, S3_IN_ONE * o.s3_slots));
    std::vector<std::string> oreg; std::vector<size_t> obytes;
    for (size_t i = 0; i < outs3.size(); ++i) {
      size_t b = outs3[i].per_sample_bytes * o.s3_slots;
      void* p; CUDA_CHECK(cudaMalloc(&p, b));
      std::string r = "s3o" + std::to_string(i) + "_" + tag;
      CUDA_CHECK(cudaIpcGetMemHandle(&h, p)); CHECK_OK(c3->RegisterCudaSharedMemory(r, h, 0, b));
      oreg.push_back(r); obytes.push_back(outs3[i].per_sample_bytes);
    }
    std::vector<int64_t> shp = {1, 3, o.s3_size, o.s3_size};
    for (int s = 0; s < o.s3_slots; ++s) {
      auto sl = std::make_unique<S3Slot>();
      CHECK_OK(tc::InferInput::Create(&sl->in, in3.name, shp, "FP32"));
      CHECK_OK(sl->in->SetSharedMemory("s3i_" + tag, S3_IN_ONE, S3_IN_ONE * s));
      for (size_t i = 0; i < outs3.size(); ++i) {
        tc::InferRequestedOutput* ro; CHECK_OK(tc::InferRequestedOutput::Create(&ro, outs3[i].name));
        CHECK_OK(ro->SetSharedMemory(oreg[i], obytes[i], obytes[i] * s));
        sl->outs.push_back(ro);
      }
      slots.push_back(std::move(sl));
    }
  }
  tc::InferOptions opt3(o.stage3);

  GPUDet* d_dets; int* d_count; GPUDet* h_dets; int* h_count;
  CUDA_CHECK(cudaMalloc(&d_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMallocHost((void**)&h_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMallocHost((void**)&h_count, sizeof(int)));
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t n_frames = g_frame_buf.size() / S1_IN;
  const double period = 1.0 / o.fps;
  double offset = 0;
  if (o.phase == "random") {
    std::mt19937 rng(o.seed * 1000003u + (unsigned)cam);
    offset = std::uniform_real_distribution<double>(0.0, period)(rng);
  }
  long fi = cam % (long)n_frames;
  std::this_thread::sleep_until(t_start);
  const auto hard_stop = t_start + std::chrono::duration_cast<clk::duration>(
      std::chrono::duration<double>(o.duration + 5.0));
  dim3 blk(16, 16);

  for (long k = 0;; ++k) {
    const double due_s = offset + (double)k * period;
    if (due_s >= o.duration) break;
    const auto due = t_start + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(due_s));
    const auto now = clk::now();
    if (now > hard_stop) break;
    if (now < due) std::this_thread::sleep_until(due);
    else if (now - due > std::chrono::milliseconds(1) && due_s >= o.warmup) st->late++;
    const bool rec = due_s >= o.warmup;

    if (!o.bls.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(s1_in, g_frame_buf.data() + fi * S1_IN, S1_IN, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (int j = 0; j < o.k; ++j) {   // the same seeded boxes as the client-driven path
        float side = (float)(IMG * (0.25 + 0.25 * u01(o.seed, cam, k, j, 1)));
        box_buf[4 * j + 0] = (float)((IMG - side) * u01(o.seed, cam, k, j, 2));
        box_buf[4 * j + 1] = (float)((IMG - side) * u01(o.seed, cam, k, j, 3));
        box_buf[4 * j + 2] = side; box_buf[4 * j + 3] = side;
      }
      CHECK_OK(bls_boxes->Reset());
      CHECK_OK(bls_boxes->AppendRaw(reinterpret_cast<const uint8_t*>(box_buf.data()), box_buf.size() * sizeof(float)));
      auto tb0 = clk::now();
      tc::InferResult* rb; CHECK_OK(c->Infer(&rb, optb, {bls_img, bls_boxes}, {bls_score})); delete rb;
      auto tb1 = clk::now();
      if (rec) {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->fast.push_back(std::chrono::duration<double, std::milli>(tb1 - due).count());
        st->t_s1 += std::chrono::duration<double, std::milli>(tb1 - tb0).count();   // the whole server call
        st->n++;
      }
      st->frames++;
      fi = (fi + 1) % n_frames;
      continue;
    }

    // stage 1: the frame arrives in the shm region, the locator runs, B2's post
    CUDA_CHECK(cudaMemcpyAsync(s1_in, g_frame_buf.data() + fi * S1_IN, S1_IN, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = clk::now();
    tc::InferResult* r1; CHECK_OK(c->Infer(&r1, opt1, {s1i}, {s1o})); delete r1;
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
        s1_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int nd = std::min(*h_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpy(h_dets, d_dets, nd * sizeof(GPUDet), cudaMemcpyDeviceToHost));
    st->dets += nms_count(h_dets, nd, 0.45f);
    auto t1 = clk::now();

    // crops: K seeded boxes, cut and resized on the GPU
    struct Box { float x, y, w, h; };
    std::vector<Box> boxes(o.k);
    dim3 g2((o.s2_size + 15) / 16, (o.s2_size + 15) / 16);
    for (int j = 0; j < o.k; ++j) {
      float side = (float)(IMG * (0.25 + 0.25 * u01(o.seed, cam, k, j, 1)));
      boxes[j] = {(float)((IMG - side) * u01(o.seed, cam, k, j, 2)),
                  (float)((IMG - side) * u01(o.seed, cam, k, j, 3)), side, side};
      crop_resize_kernel<<<g2, blk, 0, stream>>>(s1_in, IMG, IMG, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h,
                                                 s2_in + (size_t)j * 3 * o.s2_size * o.s2_size, o.s2_size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t2 = clk::now();

    // stage 2: one request of batch K
    tc::InferResult* r2; CHECK_OK(c->Infer(&r2, opt2, {s2i}, s2o)); delete r2;
    auto t3 = clk::now();
    const double fast = std::chrono::duration<double, std::milli>(t3 - due).count();

    // stage 3: seeded flags; async, never blocks the line
    if (use_s3) {
      for (int j = 0; j < o.k; ++j) {
        if (u01(o.seed, cam, k, j, 4) >= o.p) continue;
        if (rec) st->flagged++;
        if (!g_budget.take(o.s3_cost_ms)) { if (rec) st->s3_shed++; continue; }
        S3Slot* free_slot = nullptr; int si = -1;
        for (int s = 0; s < o.s3_slots; ++s) {
          bool expect = false;
          if (slots[s]->busy.compare_exchange_strong(expect, true)) { free_slot = slots[s].get(); si = s; break; }
        }
        if (!free_slot) { if (rec) st->s3_skipped++; continue; }
        dim3 g3((o.s3_size + 15) / 16, (o.s3_size + 15) / 16);
        float* dst = s3_in + (size_t)si * 3 * o.s3_size * o.s3_size;
        if (o.s3_input == "frame")
          crop_resize_kernel<<<g3, blk, 0, stream>>>(s1_in, IMG, IMG, 0, 0, IMG, IMG, dst, o.s3_size);
        else
          crop_resize_kernel<<<g3, blk, 0, stream>>>(s1_in, IMG, IMG, boxes[j].x, boxes[j].y, boxes[j].w,
                                                     boxes[j].h, dst, o.s3_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        free_slot->due = due; free_slot->record = rec;
        std::vector<const tc::InferRequestedOutput*> ro(free_slot->outs.begin(), free_slot->outs.end());
        if (rec) st->s3_sent++;
        CHECK_OK(c3->AsyncInfer(
            [free_slot, st](tc::InferResult* res) {
              double ms = std::chrono::duration<double, std::milli>(clk::now() - free_slot->due).count();
              if (free_slot->record) {
                st->s3_done++;
                std::lock_guard<std::mutex> lk(st->mtx); st->explain.push_back(ms);
              }
              delete res;
              free_slot->busy.store(false);
            },
            opt3, {free_slot->in}, ro));
      }
    }

    if (rec) {
      std::lock_guard<std::mutex> lk(st->mtx);
      st->fast.push_back(fast);
      st->t_s1 += std::chrono::duration<double, std::milli>(t1 - t0).count();
      st->t_crop += std::chrono::duration<double, std::milli>(t2 - t1).count();
      st->t_s2 += std::chrono::duration<double, std::milli>(t3 - t2).count();
      st->n++;
    }
    st->frames++;
    fi = (fi + 1) % n_frames;
  }
  // let in-flight stage-3 requests finish before the regions go away
  const auto drain = clk::now() + std::chrono::seconds(10);
  for (auto& s : slots) while (s->busy.load() && clk::now() < drain) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  // Unregister every region, after the measured window. Left registered, the
  // server keeps each run's GPU buffers mapped (CUDA IPC) after this process
  // exits, and they pile up across a sweep's runs.
  c->UnregisterCudaSharedMemory("s1i_" + tag);
  c->UnregisterCudaSharedMemory("s1o_" + tag);
  c->UnregisterCudaSharedMemory("s2i_" + tag);
  for (size_t i = 0; i < outs2.size(); ++i) c->UnregisterCudaSharedMemory("s2o" + std::to_string(i) + "_" + tag);
  if (use_s3) {
    c3->UnregisterCudaSharedMemory("s3i_" + tag);
    for (size_t i = 0; i < slots[0]->outs.size(); ++i) c3->UnregisterCudaSharedMemory("s3o" + std::to_string(i) + "_" + tag);
  }
  cudaStreamDestroy(stream);
}

static double pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  return v[std::min((size_t)(p * v.size()), v.size() - 1)];
}

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (i + 1 >= argc) break;
    if (a == "--url") o.url = nx();
    else if (a == "--stage1") o.stage1 = nx();
    else if (a == "--stage2") o.stage2 = nx();
    else if (a == "--stage3") o.stage3 = nx();
    else if (a == "--bls") o.bls = nx();
    else if (a == "--file") o.file = nx();
    else if (a == "--s3-input") o.s3_input = nx();
    else if (a == "--phase") o.phase = nx();
    else if (a == "--streams") o.streams = std::stoi(nx());
    else if (a == "--k") o.k = std::stoi(nx());
    else if (a == "--s2-size") o.s2_size = std::stoi(nx());
    else if (a == "--s3-size") o.s3_size = std::stoi(nx());
    else if (a == "--s3-slots") o.s3_slots = std::stoi(nx());
    else if (a == "--s3-budget") o.s3_budget = std::stod(nx());
    else if (a == "--s3-cost-ms") o.s3_cost_ms = std::stod(nx());
    else if (a == "--fps") o.fps = std::stod(nx());
    else if (a == "--duration") o.duration = std::stod(nx());
    else if (a == "--warmup") o.warmup = std::stod(nx());
    else if (a == "--p") o.p = std::stod(nx());
    else if (a == "--seed") o.seed = (unsigned)std::stoul(nx());
  }
  if (o.stage2.empty()) { std::cerr << "--stage2 <model> is required" << std::endl; return 2; }
  if (o.s3_budget > 0) {
    if (o.s3_cost_ms <= 0) { std::cerr << "--s3-budget needs --s3-cost-ms" << std::endl; return 2; }
    g_budget.on = true;
    g_budget.rate_ms_per_s = o.s3_budget * 1000.0;
    g_budget.cap_ms = std::max(o.s3_cost_ms, 0.1 * g_budget.rate_ms_per_s);
    g_budget.tokens_ms = g_budget.cap_ms;
  }
  if (o.s3_input != "crop" && o.s3_input != "frame") { std::cerr << "--s3-input crop|frame" << std::endl; return 2; }
  {
    std::ifstream f(o.file, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot open " << o.file << std::endl; return 2; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    g_frame_buf.resize(sz); f.read(g_frame_buf.data(), sz);
  }
  const auto t_start = clk::now() + std::chrono::duration_cast<clk::duration>(
      std::chrono::duration<double>(2.0 + 0.1 * o.streams));
  Stats st;
  std::vector<std::thread> th;
  for (int i = 0; i < o.streams; ++i) th.emplace_back(run_camera, o, i, t_start, &st);
  for (auto& t : th) t.join();
  const double elapsed = std::max(o.duration, std::chrono::duration<double>(clk::now() - t_start).count());

  std::sort(st.fast.begin(), st.fast.end());
  std::sort(st.explain.begin(), st.explain.end());
  double sn = st.n ? st.n : 1;
  printf("{\"pipeline\":\"%s\",\"stage1\":\"%s\",\"stage2\":\"%s\",\"stage3\":\"%s\","
         "\"streams\":%d,\"k\":%d,\"s2_size\":%d,\"p\":%.3f,\"s3_input\":\"%s\",\"s3_size\":%d,"
         "\"fps_per_camera\":%.1f,\"offered_fps\":%.1f,\"frames\":%ld,\"fps\":%.2f,\"late_frames\":%ld,"
         "\"fast_ms_p50\":%.3f,\"fast_ms_p95\":%.3f,\"fast_ms_p99\":%.3f,\"fast_ms_max\":%.3f,"
         "\"flagged\":%ld,\"s3_sent\":%ld,\"s3_done\":%ld,\"s3_skipped\":%ld,\"s3_shed\":%ld,\"s3_budget\":%.3f,"
         "\"explain_ms_p50\":%.3f,\"explain_ms_p95\":%.3f,\"explain_ms_p99\":%.3f,"
         "\"stages_ms\":{\"stage1\":%.3f,\"crop\":%.3f,\"stage2\":%.3f},\"seed\":%u,\"phase\":\"%s\"}\n",
         o.bls.empty() ? "inspect" : "inspect_bls", o.stage1.c_str(), o.stage2.c_str(), o.stage3.c_str(), o.streams, o.k, o.s2_size, o.p,
         o.s3_input.c_str(), o.s3_size, o.fps, o.fps * o.streams, st.frames.load(),
         st.frames.load() / elapsed, st.late.load(),
         pct(st.fast, 0.5), pct(st.fast, 0.95), pct(st.fast, 0.99), st.fast.empty() ? 0 : st.fast.back(),
         st.flagged.load(), st.s3_sent.load(), st.s3_done.load(), st.s3_skipped.load(), st.s3_shed.load(), o.s3_budget,
         pct(st.explain, 0.5), pct(st.explain, 0.95), pct(st.explain, 0.99),
         st.t_s1 / sn, st.t_crop / sn, st.t_s2 / sn, o.seed, o.phase.c_str());
  return 0;
}
