import os
from backpack import extend
from caltech_resnet18 import caltech_resnet18
from models import CNNnet, MLP, ResNet18, ResNet10, VGG11, CNNAvgPool
from ntk_active_memo_efficient import NTK_active_memo
import torch.nn as nn
from utils import LogitLoss
import torch


from torch.utils.tensorboard import SummaryWriter
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

class net(nn.Module):
    def __init__(self, dev, dim, hidden_size=100, k=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

        self.device = f'cuda:{dev}'
        os.environ['CUDA_VISIBLE_DEVICES'] = dev
        self.cuda()

        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

    def forward(self, x, dataset):
        return self.fc2(self.activate(self.fc1(x.squeeze())))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]


    def collect_grad_each_layer(self, layer_id):
        # return self.parameters()[layer_id].grad.detach().reshape(-1,)
        counter = 0
        for p in self.parameters():
            if p.requires_grad:
                if counter == layer_id:
                    return p.grad.detach().reshape(-1,)
                else:
                    counter += 1


    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)


def init_logger(args):
    # config detail to a string, for tensorboard
    if args.log_type in ['None', 'none']:
        return None, None
    if args.init_label_perC != None:
        init_label_info = 'num_perC:{}'.format(args.init_label_perC)
    elif args.init_label_num != None:
        init_label_info = 'T_num:{}'.format(args.init_label_num)
    else:
        raise NotImplementedError("Either init_label_perC or init_label_num need to be not None.")

    if len(args.query_interval) > 0:
        q_int_info = 'Qint:{}'.format(args.query_interval)
    else:
        q_int_info = 'fix_Qint:{}'.format(args.fixed_query_interval)

    active_detail_info = None
    if 'ntk' in args.active_method:
        active_detail_info = '_{}_nodeinf:{}_notrain:{}_update:{}_norm:{}_type:{}_trTime:{}'. \
            format(args.pesudo, args.wo_de_inf, args.wo_train_sim, args.grad_update_size,
                   args.norm_inf, args.inf_type, args.train_sim_times)

    if active_detail_info is not None:
        active_info = args.active_method + active_detail_info
    else:
        active_info = args.active_method

    writer_name = "{}_{}_{}_{}_perQ:{}_{}_{}_reinit:{}_lr:{}_interval:{}".format(args.dataset_str, args.base_model,
                                                                                 active_info, args.seed,
                                                                                 args.budget_num_per_query,
                                                                                 init_label_info, q_int_info,
                                                                                 args.reinit, args.lr,
                                                                                 args.fixed_query_interval)
    if args.log_type == 'tb':
        writer = SummaryWriter(comment=writer_name)
    elif args.log_type == 'txt':
        directory = parent_path + "/{}/".format(args.save_folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        writer = directory + writer_name + ".txt"
    else:
        raise NotImplementedError(f"Logger:{args.log_type} is not Implemented.")
    return writer, writer_name


def need_query(args, model, epoch, curr_round_epoch, active_times):
    better_than_prev_epochs = model.better_than_prev_epochs
    current_round_epochs = model.current_round_epochs
    train_converge_epochs = model.train_converge_epochs
    # query with fixed interval
    is_need_query = args.total_query_times > active_times
    if len(args.query_interval) > 0:
        is_need_query = is_need_query and curr_round_epoch >= args.query_interval[active_times]
        is_need_query = is_need_query and (train_converge_epochs > 10 and better_than_prev_epochs > 0 or
                                           current_round_epochs > 2 * args.query_interval[active_times])
        is_need_query = is_need_query  # and train_acc > 0.99
    elif args.fixed_query_interval > 0:
        is_need_query = is_need_query and epoch - args.starting_epoch > 0 and (
                    epoch - args.starting_epoch) % args.fixed_query_interval == 0

    return is_need_query


def epoch_check(args):
    # sanity check of the epochs and query times
    if args.total_query_times > 0:
        assert len(args.query_interval) == 0 or args.fixed_query_interval <= 0
        assert len(args.query_interval) > 0 or args.fixed_query_interval > 0

        if len(args.query_interval) > 0:
            assert len(args.query_interval) == args.total_query_times
            if args.epochs < max(args.query_interval) + args.final_extra_epochs:
                args.epochs = max(args.query_interval) + args.final_extra_epochs
        if args.fixed_query_interval > 0:
            if args.epochs < args.total_query_times * args.fixed_query_interval + args.starting_epoch + args.final_extra_epochs:
                args.epochs = args.total_query_times * args.fixed_query_interval + args.starting_epoch + args.final_extra_epochs
        print("[===== Total Epochs:{} =====]".format(args.epochs))
    else:
        args.final_extra_epochs = args.final_extra_epochs + args.epochs

    return args


def init_active_method(args, model, **kwargs):
    if args.active_method == 'ntk_memo':
        active_policy = NTK_active_memo(args)
        model = extend(model)
        model.compute_MSELoss = extend(model.compute_MSELoss)
        model.compute_CELoss = extend(model.compute_CELoss)
        model.compute_LogitLoss = extend(model.compute_LogitLoss)
    else:
        raise NotImplementedError(f"active strategy:{args.active_method} is not Implemented.")
    return active_policy


def init_model(args, train_data=None):
    if args.base_model == 'cnn':
        model = CNNnet()
    elif args.base_model == 'cnn_avgpool':
        model = CNNAvgPool()
    elif args.base_model == 'mlp':
        model = MLP(args, args.input_dim, args.class_num)
    elif args.base_model == 'resnet':
        if args.dataset_str == 'caltech101':
            model = caltech_resnet18(args.class_num)
        else:
            input_dim = train_data[0][0].shape[0]
            k = len(set(train_data[1]))
            model = net('0', input_dim, k=k) #ResNet18()
    elif args.base_model == 'vgg':
        model = VGG11()
    else:
        raise NotImplementedError(f"(Dataset:{args.dataset_str}) Model:{args.base_model} is not Implemented.")

    model.MSE = args.MSE
    model.best_test_acc = 0
    model.prev_best_test_acc = 0
    model.better_than_prev_epochs = 0
    model.current_round_epochs = 0
    model.train_converge_epochs = 0
    print("\n# params:{}.\n".format(model.count_parameters()))

    if args.cuda:
        model.device = 'cuda'
        model.cuda()
    else:
        model.device = 'cpu'

    return model
