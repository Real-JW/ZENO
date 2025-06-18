rustup override set 1.43.0
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin shallownet_knit_encoding --release > result/shallownet_knit.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin lenet_cifar_small_knit_encoding --release > result/lenet_cifar_small.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin lenet_cifar_large_knit_encoding --release > result/lenet_cifar_large.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin vgg_one_private --release > result/vgg_one_private.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin vgg11_one_private --release > result/vgg11_one_private.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin resnet18_one_private --release > result/resnet18_one_private.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin vgg_both_private --release > result/vgg_both_private.log

# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin baseline_lenet_small --release > result/baseline_lenet_small.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin baseline_lenet_large --release > result/baseline_lenet_large.log
# CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin baseline_vgg16 --release > result/baseline_vgg16.log
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --bin baseline_resnet18 --release > result/baseline_resnet18.log
