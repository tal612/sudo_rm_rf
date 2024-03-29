dsi_gpu=$1
n_epochs=$2
modell=$3
wd=$4
#path="/home/dsi/yechezo/sudo_rm_rf_/weights/new/"
checkpoints_path_="/home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/dnn/experiments/weights/"

#echo $checkpoints_path
screen -S $n_epochs
eval "$(conda shell.bash hook)"
conda activate sudo_rm_rf
cd /home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/dnn/experiments
python /home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/dnn/experiments/run_gender_detector.py --dsi_gpu $dsi_gpu --train WHAMR --separation_task noisy_reverberant -bs 1 --max_num_sources 2 --train_val WHAMR --checkpoints_path $checkpoints_path_ --n_epochs $n_epochs --save_checkpoint_every 1 --save_best_weights True --model modell --weight_decay wd

#                                                                                                          --train WHAMR --separation_task noisy_reverberant -bs 1 --max_num_sources 2 --train_val WHAMR --checkpoints_path /home/dsi/yechezo/sudo_rm_rf_/weights/examination --n_epochs 3 --save_checkpoint_every 1 --save_best_weights True
