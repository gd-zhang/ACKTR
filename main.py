import tensorflow as tf
from utils.utils import create_experiment_dirs
from utils.utils import parse_args
from ACKTR import ACKTR
import logger


def main():
    # Parse the JSON arguments
    config_args = None
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=config_args.num_envs,
                            inter_op_parallelism_threads=config_args.num_envs)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare Directories
    # TODO: add openai logger
    config_args.experiment_dir, config_args.summary_dir, config_args.checkpoint_dir, config_args.output_dir, config_args.test_dir = \
        create_experiment_dirs(config_args.experiment_dir)
    logger.configure(config_args.experiment_dir)
    logger.info("Print configuration .....")
    logger.info(config_args)

    acktr = ACKTR(sess, config_args)

    if config_args.to_train:
        acktr.train()
    if config_args.to_test:
        acktr.test(total_timesteps=10000000)


if __name__ == '__main__':
    main()
