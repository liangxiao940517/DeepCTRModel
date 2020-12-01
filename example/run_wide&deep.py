from tensorflow.python.keras.callbacks import EarlyStopping

from src.model.MLP import MLP
from src.model.WideDeep import WideDeep
from src.generator.CriteoGenerator import CriteoGenerator
from conf.CommonConfig import CommonConfig as cf
from conf.ModelConfig import ModelConfig as mf



if __name__ == "__main__":
    file_path = cf.DATA_DIR + cf.DATA_FILE

    dataset_generator = CriteoGenerator(file_path)

    feature_columns, train, test = dataset_generator.generateCriteoData()
    train_X, train_Y = train
    test_X, test_Y = test

    model = WideDeep(feature_columns=feature_columns,hidden_units=mf.DEEP_HIDDEN_UNITS,activation=mf.ACTIVATION)
    model.compile(optimizer=mf.OPTIMIZER,loss=mf.LOSS_FUNCTION,metrics=mf.METRIC)

    model.fit(train_X,
              train_Y,
              epochs=mf.EPOCHS,
              callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],
              batch_size=mf.BATCH_SIZE,
              validation_split=mf.VALIDATION_RATIO)

    #print('test AUC: %f' % model.evaluate(test_X, test_Y))
    print(model.evaluate(test_X, test_Y))

    pre_test_Y = model.predict(test_X)
    print(pre_test_Y)