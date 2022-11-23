from utils import get_data,train
from module import get_backbone,attch_projection_head
import argparse
parser = argparse.ArgumentParser(description='training setup')

parser.add_argument('--batch_size', type=int, default=1024, help='batch size of training')
parser.add_argument('--epoch', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--tem', type=float, default=1e-1, help='the hyperparameter temperature')
parser.add_argument('--dataset', type=str, default='ucihar', choices=['ucihar', 'motion', 'uschad'], help='dataset')
parser.add_argument('--backbone', type=str, default='tpn', help='backbone')
parser.add_argument('--p1', type=int, default=96, help='projection head dimension')
parser.add_argument('--p2', type=int, default=96, help='projection head dimension')
parser.add_argument('--p3', type=int, default=96, help='projection head dimension')
parser.add_argument('--cluster', type=str, default='birch', choices=['birch', 'kmeans'], help='cluster methods')
parser.add_argument('--cluster_num', type=int, default=6, help='cluster number')

if __name__ == '__main__':
    args = parser.parse_args()

    x_data,y_data = get_data(args.dataset)
    n_timesteps, n_features, n_outputs = x_data.shape[1], x_data.shape[2], y_data.shape[1]

    backbone = get_backbone(args.backbone,n_timesteps,n_features)
    model = attch_projection_head(backbone,args.p1,args.p2,args.p3)

    train(model,x_data,args)