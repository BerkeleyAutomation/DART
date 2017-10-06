python tests/bc.py --envname Hopper-v1 --t 500 --iters 5 10 15 20
python tests/dart.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --update 1 7
python tests/rand.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --prior 1.0
python tests/dagger.py --envname Hopper-v1 --t 500 --iters 5 10 15 20 --beta .5

# python test_bc.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20
# python test_dart.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --update 1 7
# python test_rand.py --envname Walker2d-v1 --t 500 --iters 5 10 15 20 --prior 1.0

# python test_bc.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20
# python test_dart.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --update 1 7
# python test_rand.py --envname HalfCheetah-v1 --t 500 --iters 5 10 15 20 --prior 1.0
