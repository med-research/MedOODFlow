import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config, timing_wrapper=True)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)
        if self.config.evaluator.ood_scheme == 'fsood':
            acc_metrics = evaluator.eval_acc(
                net,
                id_loader_dict['test'],
                postprocessor.postprocessor,  # no timing wrapper
                fsood=True,
                csid_data_loaders=ood_loader_dict['csid'])
        else:
            acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                             postprocessor.postprocessor)
        print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        eval_ood_time = time.time() - timer
        timing_stats = postprocessor.get_timing_stats()
        print('\nPostprocessor Timing Stats:', flush=True)
        for k, v in timing_stats.items():
            print(f'  {k}: {v}', flush=True)
        if hasattr(evaluator, 'get_timing_stats'):
            timing_stats = evaluator.get_timing_stats()
            timing_stats['eval_ood Time (s)'] = round(eval_ood_time, 3)
            print('\nEvaluator Timing Stats:', flush=True)
            for k, v in timing_stats.items():
                print(f'  {k}: {v}', flush=True)
        else:
            print('Time used for eval_ood: {:.0f}s'.format(eval_ood_time))
        print('Completed!', flush=True)
