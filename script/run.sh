export PYTHONPATH=/jiangjiajun/github/wangjunjie06/tvm/python
export CUDA_VISIBLE_DEVICES='5' 
echo "start  autoschecule gpu"
echo `date`
python tune_network.py --network DBFace_20000_pretrained --n-trials 20000 --cost-model xgb --load-model /jiangjiajun/github/wangjunjie06/tenset/scripts/xgb_k80_t4_all.pkl --target cuda --relay-file relay_files/DBFace/relay.pkl 1>as_gpu_DBFace_20000_pretrained.out 2>as_gpu_DBFace_20000_pretrained.err
mv total_latency.tsv total_latency.DBFace_20000_pretrained.tsv
rm -rf xgb_k80_t4_all.pkl
cp ../xgb_k80_t4_all.pkl .

python tune_network.py --network Ultra_1MB_20000_pretrained_new --n-trials 20000 --cost-model xgb --load-model /jiangjiajun/github/wangjunjie06/tenset/scripts/xgb_k80_t4_all.pkl --target cuda --relay-file relay_files/ultra_1mb/relay.pkl 1>as_gpu_ultra_1mb_20000_pretrained_new.out 2>as_gpu_ultra_1mb_20000_pretrained_new.err
mv total_latency.tsv total_latency.ultra_1mb_20000_pretrained_new.tsv
rm -rf xgb_k80_t4_all.pkl
cp ../xgb_k80_t4_all.pkl .

python tune_network.py --network SwinTransformer_20000_pretrained --n-trials 20000 --cost-model xgb --load-model /jiangjiajun/github/wangjunjie06/tenset/scripts/xgb_k80_t4_all.pkl --target cuda --relay-file relay_files/SwinTransformer/relay.pkl 1>as_gpu_SwinTransformer_20000_pretrained.out 2>as_gpu_SwinTransformer_20000_pretrained.err
mv total_latency.tsv total_latency.SwinTransformer_20000_pretrained.tsv
rm -rf xgb_k80_t4_all.pkl
cp ../xgb_k80_t4_all.pkl .

python tune_network.py --network squeezenet_20000_pretrained --n-trials 20000 --cost-model xgb --load-model /jiangjiajun/github/wangjunjie06/tenset/scripts/xgb_k80_t4_all.pkl --target cuda --relay-file relay_files/squeezenet/relay.pkl 1>as_gpu_squeezenet_20000_pretrained.out 2>as_gpu_squeezenet_20000_pretrained.err
mv total_latency.tsv total_latency.squeezenet_20000_pretrained.tsv
rm -rf xgb_k80_t4_all.pkl
cp ../xgb_k80_t4_all.pkl .

# export CUDA_VISIBLE_DEVICES='' 
# echo "start  autoschecule cpu"
# echo `date`
# python tune_network.py --network DBFace_6000 --n-trials 6000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=core-avx2" --relay-file relay_files/DBFace/relay.pkl 1>as_cpu_DBFace_6000.out 2>as_cpu_DBFace_6000.err
# mv total_latency.tsv total_latency.DBFace_6000.cpu.tsv
# python tune_network.py --network DBFace_3000 --n-trials 3000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/DBFace/relay.pkl 1>as_cpu_DBFace_3000.out 2>as_cpu_DBFace_6000.err
# mv total_latency.tsv total_latency.DBFace_3000.cpu.tsv
# python tune_network.py --network DBFace_1200 --n-trials 1200 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/DBFace/relay.pkl 1>as_cpu_DBFace_1200.out 2>as_cpu_DBFace_1200.err
# mv total_latency.tsv total_latency.DBFace_1200.cpu.tsv
# python tune_network.py --network DBFace_400 --n-trials 400 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/DBFace/relay.pkl 1>as_cpu_DBFace_400.out 2>as_cpu_DBFace_400.err
# mv total_latency.tsv total_latency.DBFace_400.cpu.tsv

# python tune_network.py --network Ultra_1MB_6000 --n-trials 6000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/ultra_1mb/relay.pkl 1>as_cpu_ultra_1mb_6000.out 2>as_cpu_ultra_1mb_6000.err
# mv total_latency.tsv total_latency.ultra_1mb_6000.cpu.tsv
# python tune_network.py --network Ultra_1MB_3000 --n-trials 3000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/ultra_1mb/relay.pkl 1>as_cpu_ultra_1mb_3000.out 2>as_cpu_ultra_1mb_3000.err
# mv total_latency.tsv total_latency.ultra_1mb_3000.cpu.tsv
# python tune_network.py --network Ultra_1MB_1200 --n-trials 1200 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/ultra_1mb/relay.pkl 1>as_cpu_ultra_1mb_1200.out 2>as_cpu_ultra_1mb_1200.err
# mv total_latency.tsv total_latency.ultra_1mb_1200.cpu.tsv
# python tune_network.py --network Ultra_1MB_400 --n-trials 400 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/ultra_1mb/relay.pkl 1>as_cpu_ultra_1mb_400.out 2>as_cpu_ultra_1mb_400.err
# mv total_latency.tsv total_latency.ultra_1mb_400.cpu.tsv

# python tune_network.py --network SwinTransformer_6000 --n-trials 6000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/SwinTransformer/relay.pkl 1>as_cpu_SwinTransformer_6000.out 2>as_cpu_SwinTransformer_6000.err
# mv total_latency.tsv total_latency.SwinTransformer_6000.cpu.tsv
# python tune_network.py --network SwinTransformer_3000 --n-trials 3000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/SwinTransformer/relay.pkl 1>as_cpu_SwinTransformer_3000.out 2>as_cpu_SwinTransformer_3000.err
# mv total_latency.tsv total_latency.SwinTransformer_3000.cpu.tsv
# python tune_network.py --network SwinTransformer_1200 --n-trials 1200 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/SwinTransformer/relay.pkl 1>as_cpu_SwinTransformer_1200.out 2>as_cpu_SwinTransformer_1200.err
# mv total_latency.tsv total_latency.SwinTransformer_1200.cpu.tsv
# python tune_network.py --network SwinTransformer_400 --n-trials 400 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/SwinTransformer/relay.pkl 1>as_cpu_SwinTransformer_400.out 2>as_cpu_SwinTransformer_400.err
# mv total_latency.tsv total_latency.SwinTransformer_400.cpu.tsv

# python tune_network.py --network squeezenet_6000 --n-trials 6000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/squeezenet/relay.pkl 1>as_cpu_squeezenet_6000.out 2>as_cpu_squeezenet_6000.err
# mv total_latency.tsv total_latency.squeezenet_6000.cpu.tsv
# python tune_network.py --network squeezenet_3000 --n-trials 3000 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/squeezenet/relay.pkl 1>as_cpu_squeezenet_3000.out 2>as_cpu_squeezenet_3000.err
# mv total_latency.tsv total_latency.squeezenet_3000.cpu.tsv
# python tune_network.py --network squeezenet_1200 --n-trials 1200 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/squeezenet/relay.pkl 1>as_cpu_squeezenet_1200.out 2>as_cpu_squeezenet_1200.err
# mv total_latency.tsv total_latency.squeezenet_1200.cpu.tsv
# python tune_network.py --network squeezenet_400 --n-trials 400 --cost-model xgb-no-update --load-model xgb_cpu_platinum-8272.pkl --target "llvm -mcpu=skylake-avx512" --relay-file relay_files/squeezenet/relay.pkl 1>as_cpu_squeezenet_400.out 2>as_cpu_squeezenet_400.err
# mv total_latency.tsv total_latency.squeezenet_400.cpu.tsv

#python scripts/auto_schedule_cpu.py 1>as_cpu.out 2>as_cpu.err
#mv total_latency.tsv total_latency.cpu.tsv 
#echo "start autoschedule gpu"
#echo `date`
#python scripts/auto_schedule_gpu.py 1>as_gpu.out 2>as_gpu.err

