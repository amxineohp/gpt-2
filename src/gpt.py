import logging, sys, os, inspect, json, numpy as np
from copy import deepcopy
import tensorflow as tf
import mautil as mu
from mautil.basic_model import InputFeature
from mautil import data_reader
from mautil.tf_models import TF
import model, sample, encoder
import util

SEED = 9527
gl = globals()


class GPT(TF):
    cfg = deepcopy(TF.cfg)
    cfg.batch_reader = 'SeqBatchReader'
    cfg.batch_reader_cfg = {'process_mode': 'N', 'max_seq_len': 512}
    cfg.models_dir = 'models'
    cfg.model_name = '117M'
    cfg.use_past = False

    def __init__(self, name, cfg={}, batch_reader=None):
        self.cfg = deepcopy(self.cfg)
        self.cfg.update(cfg)
        cfg = self.cfg

        self.enc = encoder.get_encoder(cfg.model_name, cfg.models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(cfg.models_dir, cfg.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        if cfg.debug:
            cfg.batch_reader_cfg['max_seq_len'] = 2
            hparams.n_layer=2
            hparams.n_layer=2
            hparams.n_head=2
            hparams.n_embd=8
            hparams.n_ctx=16
            cfg.batch_reader_cfg['max_seq_len'] = 8
        cfg.hparams = hparams
        self.past = None

        super(GPT, self).__init__(name, {}, batch_reader)


    def _process_data(self, data, data_type):
        x = {'fnames':[], 'seqs':[]}
        for fname, text in data.items():
            seq = np.stack(self.enc.encode(text)).astype(np.int32)
            x['fnames'].append(fname)
            x['seqs'].append(seq)
        return x, None

    def fit_batch(self, batch):
        if self.past is not None:
            batch['past'] = self.past
        outs = super(GPT, self).fit_batch(batch)
        return outs

    def _add_train_nodes(self):
        super(GPT, self)._add_train_nodes()
        if self.cfg.use_past:
            self.train_nodes['present'] = self.lm_output['present']
            self.validate_nodes['present'] = self.lm_output['present']


    def _add_loss(self):
        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.seqs[:, 1:], \
                                                                                   logits=self.lm_output['logits'][:, :-1]))

    def _add_main_graph(self):
        self.lm_output = model.model(hparams=self.cfg.hparams, X=self.seqs, past=self.past, reuse=tf.AUTO_REUSE)


    def _init_input_features(self):
        features = []
        features.append(InputFeature('seqs', [None, None], tf.int32))
        if self.cfg.use_past:
            features.append(InputFeature('past', [None, self.cfg.hparams.n_layer, 2, self.cfg.hparams.n_head,  None, self.cfg.hparams.n_embd], tf.float32))
        return features


class ArgParser(mu.TrainArgParser):
    @staticmethod
    def add_args(parser):
        super(ArgParser, ArgParser).add_args(parser)
        parser.add_argument("-use_past", "--use_past", action="store_true", help="use past")
        parser.add_argument("-model_name", "--model_name", help="model name:117M")


def train(args):
    trainer = mu.training.Trainer('Trainer', SEED)
    data = util.load_data(args.dataset, dir=args.dataset, debug=args.debug)

    trainer.train_model(data, args, gl)


if __name__ == '__main__':
    args = ArgParser.load_args()

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')

    if args.method_or_model in gl:
        if inspect.isfunction(gl[args.method_or_model]):
            gl[args.model_name]()
        else:
            args.model_names = args.method_or_model
            train(args)
    else:
        logging.error('unknown method or model: %s', args.method_or_model)

