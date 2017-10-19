python test_bc.py --envname Hopper-v1 --t 500 --iters 5 10 15 20
python test_dart.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7
python test_rand.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --prior 1.0
python test_dagger.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --beta .5
python test_dagger_b.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7 --beta .5
python test_iso.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --prior 1.0

python test_bc.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20
python test_dart.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7
python test_rand.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --prior 1.0
python test_dagger.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --beta .5
python test_dagger_b.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7 --beta .5
python test_iso.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --prior 1.0

python test_bc.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20
python test_dart.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7
python test_rand.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --prior 1.0
python test_dagger.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --beta .5
python test_dagger_b.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7 --beta .5
python test_iso.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --prior 1.0

python test_bc.py --envname Humanoid-v1 --t 500 --iters 50 100 150 200
python test_dart.py --envname Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195
python test_rand.py --envname Humanoid-v1 --t 500 --iters 50 100 150 200 --prior 1.0
python test_dagger --envname Humanoid-v1 --t 500 --iters 50 100 150 200 --beta .5
python test_dagger_b.py --envname Humanoid-v1 --t 500 --iters 50 100 150 200 --update 60 75 90 105 120 135 150 165 180 195 --beta .5
python test_iso.py --envname Humanoid-v1 --t 500 --iters 50 100 150 200 --prior 1.0

