"""Phase 4: the inspection fast path as ONE Triton request (BLS, Python backend).

The client-driven line (Phase 2) makes two gRPC calls per frame (stage 1, then
stage 2) and does stage 1's post-processing and the crops itself. Here the
client sends one request carrying the frame (CUDA shared memory) and its K crop
boxes, and this model does the rest inside the server, on the GPU:

  stage 1   BLS call to yolov8s
  post      the same candidate filter (score > 0.25) and class-aware NMS
            (IoU 0.45) as flow B2, but on the GPU with torch/torchvision
  crops     torchvision roi_align, one bilinear sample per output pixel
            (sampling_ratio=1, aligned=True): the same resize as the client's
            crop kernel
  stage 2   BLS call to s2 with the crops as a GPU tensor (DLPack, no copy)

It returns the K image scores and the stage-1 detection count. Stage 3 is not
part of this arm: Phase 4 compares the fast path only.
"""
import numpy as np
import torch
import torchvision
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:
    def initialize(self, args):
        self.dev = torch.device("cuda")
        self.size = 256

    def execute(self, requests):
        responses = []
        for req in requests:
            frame = pb_utils.get_input_tensor_by_name(req, "images")        # 1x3x640x640, GPU
            boxes = pb_utils.get_input_tensor_by_name(req, "boxes")         # Kx4 (x, y, w, h)

            # stage 1
            r1 = pb_utils.InferenceRequest(model_name="yolov8s", inputs=[frame],
                                           requested_output_names=["output0"]).exec()
            if r1.has_error():
                raise pb_utils.TritonModelException(r1.error().message())
            out = from_dlpack(pb_utils.get_output_tensor_by_name(r1, "output0").to_dlpack())
            o = out[0].to(self.dev)                                          # 84 x 8400
            scores, cls = o[4:].max(0)
            keep = scores > 0.25
            b = o[:4, keep].t()
            xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
            ndet = torchvision.ops.batched_nms(xyxy, scores[keep], cls[keep], 0.45).numel()

            # crops, on the GPU
            img = from_dlpack(frame.to_dlpack()).to(self.dev)
            bx = torch.as_tensor(boxes.as_numpy() if boxes.is_cpu() else from_dlpack(boxes.to_dlpack()),
                                 device=self.dev, dtype=torch.float32)
            rois = torch.cat([torch.zeros(bx.shape[0], 1, device=self.dev),
                              bx[:, :1], bx[:, 1:2], bx[:, :1] + bx[:, 2:3], bx[:, 1:2] + bx[:, 3:4]], 1)
            crops = torchvision.ops.roi_align(img, rois, output_size=self.size, spatial_scale=1.0,
                                              sampling_ratio=1, aligned=True).contiguous()

            # stage 2
            r2 = pb_utils.InferenceRequest(
                model_name="s2", inputs=[pb_utils.Tensor.from_dlpack("images", to_dlpack(crops))],
                requested_output_names=["score"]).exec()
            if r2.has_error():
                raise pb_utils.TritonModelException(r2.error().message())
            score = pb_utils.get_output_tensor_by_name(r2, "score")
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack("score", score.to_dlpack()),
                pb_utils.Tensor("ndet", np.array([ndet], dtype=np.int32))]))
        return responses
