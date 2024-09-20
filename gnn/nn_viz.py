

import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

#
#
# embedding_h.weight
# embedding_h.bias
# embedding_e.weight
# embedding_e.bias
# GatedGCN_layers.0.A.weight
# GatedGCN_layers.0.A.bias
# GatedGCN_layers.0.B.weight
# GatedGCN_layers.0.B.bias
# GatedGCN_layers.0.C.weight
# GatedGCN_layers.0.C.bias
# GatedGCN_layers.0.D.weight
# GatedGCN_layers.0.D.bias
# GatedGCN_layers.0.E.weight
# GatedGCN_layers.0.E.bias
# GatedGCN_layers.0.bn_node_h.weight
# GatedGCN_layers.0.bn_node_h.bias
# GatedGCN_layers.0.bn_node_e.weight
# GatedGCN_layers.0.bn_node_e.bias
# MLP_layer.FC_layers.0.weight
# MLP_layer.FC_layers.0.bias
# MLP_layer.FC_layers.1.weight
# MLP_layer.FC_layers.1.bias
# MLP_layer.FC_layers.2.weight
# MLP_layer.FC_layers.2.bias
#
