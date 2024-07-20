from cnnClassifier import logger
from cnnClassifier.components.model_trainer import Training
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME="Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == "main":
    try:
        logger.info(f"******")
        logger.info(f">>>Stage {STAGE_NAME} started")
        obj=ModelTrainingPipeline()
        obj.main()
    except Exception as e:
         logger.exception(e)
         raise e
