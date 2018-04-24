python experiments/plot_loss.py --env Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
python experiments/plot_reward.py --env Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save --normalize
# python experiments/plot_data.py --env Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_data2.py --env Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_loss_rand.py --env Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save

python experiments/plot_loss.py --env Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
python experiments/plot_reward.py --env Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save --normalize
# python experiments/plot_data.py --env Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_data2.py --env Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_loss_rand.py --env Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
 
python experiments/plot_loss.py --env HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
python experiments/plot_reward.py --env HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save --normalize
# python experiments/plot_loss_rand.py --env HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_data.py --env HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save
# python experiments/plot_data2.py --env HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --save

python experiments/plot_loss.py --env Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --save
python experiments/plot_reward.py --env Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --normalize --save
# # python experiments/plot_data.py --env Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --normalize --save
# python experiments/plot_data2.py --env Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --normalize --save
# python experiments/plot_loss_rand.py --env Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --save
