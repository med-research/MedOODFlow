import time

import torch


class PostprocessorTimingWrapper:
    """Wrapper to track timing for postprocessor methods."""
    def __init__(self, postprocessor):
        self.postprocessor = postprocessor
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.num_inference_samples = 0
        self.num_postprocess_samples = 0
        self.setup_time = 0.0

    def setup(self, *args, **kwargs):
        start = time.time()
        result = self.postprocessor.setup(*args, **kwargs)
        self.setup_time += time.time() - start
        return result

    def inference(self, net, data_loader):
        start = time.time()
        pred, conf, gt = self.postprocessor.inference(net, data_loader)
        self.inference_time += time.time() - start
        num_samples = pred.size(0) if torch.is_tensor(pred) else len(pred)
        self.num_inference_samples += num_samples
        return pred, conf, gt

    def postprocess(self, net, data):
        start = time.time()
        result = self.postprocessor.postprocess(net, data)
        elapsed = time.time() - start
        self.postprocess_time += elapsed
        batch_size = data.size(0) if torch.is_tensor(data) else len(data)
        self.num_postprocess_samples += batch_size
        return result

    def get_timing_stats(self):
        postprocess_time = self.postprocess_time + \
            getattr(self.postprocessor, 'postprocess_time', 0.0)
        num_postprocess_samples = self.num_postprocess_samples + \
            getattr(self.postprocessor, 'num_postprocess_samples', 0)
        avg_inference_ms = \
            (self.inference_time / self.num_inference_samples * 1000) \
            if self.num_inference_samples > 0 else 0
        avg_postprocess_ms = \
            (postprocess_time / num_postprocess_samples * 1000) \
            if num_postprocess_samples > 0 else 0

        return {
            'Setup Time (s)':
            round(self.setup_time, 3),
            'Total Inference Time (s)':
            round(self.inference_time, 3),
            'Total Postprocess Time (s)':
            round(postprocess_time, 3),
            'Number of Inference Samples':
            self.num_inference_samples,
            'Number of Postprocess Samples':
            num_postprocess_samples,
            'Average Inference Time per Sample (ms)':
            round(avg_inference_ms, 3),
            'Average Postprocess Time per Sample (ms)':
            round(avg_postprocess_ms, 3)
        }

    def __getattr__(self, name):
        return getattr(self.postprocessor, name)
