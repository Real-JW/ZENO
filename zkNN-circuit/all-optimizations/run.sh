# Cap CPU0 to 600000 kHz and set governor to userspace
sudo cpufreq-set -c 0 -g userspace
sudo cpufreq-set -c 0 -f 600000

# Disable all other cores except core0
echo 0 | sudo tee /sys/devices/system/cpu/cpu1/online
if [ -e /sys/devices/system/cpu/cpu2/online ]; then echo 0 | sudo tee /sys/devices/system/cpu/cpu2/online; fi
if [ -e /sys/devices/system/cpu/cpu3/online ]; then echo 0 | sudo tee /sys/devices/system/cpu/cpu3/online; fi

# Pin all processes to core0
rustup override set 1.43.0
# taskset -c 0 cargo run --bin baseline_lenet_small --release > result/baseline_lenet_small.log
# taskset -c 0 cargo run --bin baseline_lenet_large --release > result/baseline_lenet_large.log
# taskset -c 0 cargo run --bin baseline_vgg16 --release > result/baseline_vgg16.log
taskset -c 0 cargo run --bin baseline_resnet18 --release > result/baseline_resnet18.log
