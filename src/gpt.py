import logging, sys, os, inspect, json, numpy as np
from copy import deepcopy
from collections import OrderedDict
import tensorflow as tf
import mautil as mu
from mautil.basic_model import InputFeature
from mautil import data_reader
from mautil.tf_models import TF
from tensorflow.core.protobuf import rewriter_config_pb2
import model, sample, encoder
import util

SEED = 9527
gl = globals()


class GPT(TF):
    cfg = deepcopy(TF.cfg)
    cfg.batch_reader = 'SeqBatchReader'
    cfg.restore_openai = False
    cfg.openai_model = '117M'
    cfg.use_past = False
    cfg.n_vocab = 50257
    cfg.n_ctx = 1024
    cfg.n_embd = 768
    cfg.n_head = 12
    cfg.n_layer = 12
    cfg.save_processed = False
    cfg.batch_reader_cfg = {'process_mode': 'N'}

    def __init__(self, name, cfg={}, batch_reader=None):
        self.cfg = deepcopy(self.cfg)
        self.cfg.update(cfg)
        cfg = self.cfg
        n_ctx = cfg.n_ctx

        self.enc = encoder.get_encoder(cfg.openai_model, os.path.join(cfg.data_dir, 'models'))
        with open(os.path.join(cfg.data_dir, 'models', cfg.openai_model, 'hparams.json')) as f:
            cfg.update(json.load(f))
        if cfg.debug:
            cfg.n_layer=2
            cfg.n_head=2
            cfg.n_embd=8
            cfg.n_ctx=16
            cfg.batch_reader_cfg['max_seq_len'] = 8
        cfg.n_ctx = n_ctx
        cfg.batch_reader_cfg['max_seq_len'] = cfg.n_ctx
        if cfg.use_past:
            assert cfg.batch_size==1, "current only batch size 1 supported when use past"
            cfg.val_batch_size = cfg.batch_size
            cfg.batch_reader_cfg['ordered'] = True
            cfg.batch_reader_cfg['max_seq_len'] = cfg.n_ctx//2

        self.present = None
        self.past = None

        super(GPT, self).__init__(name, {}, batch_reader)
        super(TF, self).save()

        if cfg.restore_openai:
            self.create_model()
            with self._graph.as_default():
                self.restore(model_dir= os.path.join(cfg.data_dir, 'models', cfg.openai_model), var_list=tf.trainable_variables())

    def _process_data(self, data, data_type):
        x = {'fnames':[], 'seqs':[]}
        logging.info('start process data %s', data_type)

        processed_fname = self.gen_fname('', self.cfg.dataset + '_' + data_type + '_processed.dump')
        if self.cfg.save_processed:
            if os.path.exists(processed_fname):
                logging.info('load processed data from %s', processed_fname)
                x = mu.load_dump(processed_fname)
                return x, None
        for fname, text in data.items():
            seqs = []; chunk = 10000000
            for i in range((len(text) + chunk -1)//chunk):
                seq = np.array(self.enc.encode(text[i*chunk:(i+1)*chunk]))
                seqs.append(seq)
            seq = np.concatenate(seqs)
            x['fnames'].append(fname)
            x['seqs'].append(seq)
        if self.cfg.save_processed:
            mu.dump(x, processed_fname)

        logging.info('total seqs %s, total tokens %s', len(x['seqs']), sum(map(len, x['seqs'])))
        return x, None

    def pre_run_batch(self, batch, epoch=0, itr=0, global_step=0, is_training=True):
        batch = super(GPT, self).pre_run_batch(batch, epoch, itr, global_step, is_training)
        if self.cfg.use_past:
            if self.present is None:
                self.present = batch['past'] = self._rs.rand(batch['batch_size'], self.cfg.n_layer, 2, self.cfg.n_head, self.cfg.n_ctx//2, self.cfg.n_embd//self.cfg.n_head)
            batch['past'] = self.present
        return batch

    def run(self, sess, batch, nodes):
        feed = self.get_feed(batch)
        outputs = sess.run(nodes, feed)
        if self.cfg.use_past:
            self.present = outputs.pop('present')
        return outputs

    def _add_train_nodes(self):
        super(GPT, self)._add_train_nodes()
        if self.cfg.use_past:
            self.train_nodes['present'] = self.lm_output['present']
            self.validate_nodes['present'] = self.lm_output['present']

    def _add_loss(self):
        weights = tf.sequence_mask(self.seqs_len-1, tf.reduce_max(tf.shape(self.seqs)[-1])-1, dtype=tf.float32)
        self._loss = tf.reduce_sum(weights*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.seqs[:, 1:], \
                                  logits=self.lm_output['logits'][:, :-1])) / (tf.reduce_sum(weights) + 1e-15)

    def _add_main_graph(self):
        self.lm_output = model.model(hparams=self.cfg, X=self.seqs, past=self.past, reuse=tf.AUTO_REUSE)

    def _init_input_features(self):
        features = []
        features.append(InputFeature('seqs', [None, None], tf.int32))
        features.append(InputFeature('seqs_len', [None], tf.int32))
        if self.cfg.use_past:
            features.append(InputFeature('past', [None, self.cfg.n_layer, 2, self.cfg.n_head,  None, self.cfg.n_embd//self.cfg.n_head], tf.float32))
        return features


class ArgParser(mu.TrainArgParser):
    @staticmethod
    def add_args(parser):
        super(ArgParser, ArgParser).add_args(parser)
        parser.add_argument("-restore_openai", "--restore_openai", action="store_true", help="restore from open ai")
        parser.add_argument("-openai_model", "--openai_model", default='117M', help="openai model used for tuning")
        parser.add_argument("-use_past", "--use_past", action="store_true", help="use_past")
        parser.add_argument("-n_ctx", "--n_ctx", type=int, help="n_ctx")



def train(args):
    trainer = mu.training.Trainer('Trainer', SEED)
    data = util.load_data(args.dataset, debug=args.debug)

    trainer.train_model(data, args, gl)


if __name__ == '__main__':
    args = ArgParser.load_args()
    from imp import reload
    reload(logging)

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')

    if args.method_or_model in gl:
        if inspect.isfunction(gl[args.method_or_model]):
            gl[args.method_or_model](args)
        else:
            args.model_names = args.method_or_model
            train(args)
    else:
        logging.error('unknown method or model: %s', args.method_or_model)

