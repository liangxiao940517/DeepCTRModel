from src.generator.AmazonGenerator import AmazonGerator
from src.model.DIN import DIN
from conf.DINModelConfig import DINModelConfig as dinc

if __name__ == "__main__":
    file_path = dinc.DATA_PATH
    dataset_generator = AmazonGerator(file_path)

    feature_columns, behavior_list, train, val, test = dataset_generator.generateAmazonData()

    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test

    model = DIN(feature_columns=feature_columns,
                behavior_feature_list=behavior_list,
                attention_hidden_units=dinc.ATTENTION_HIDDEN_UNITS,
                ffn_hidden_units=dinc.FFN_HIDDEN_UNITS,
                attention_activation=dinc.ATTENTION_ACTIVATION,
                ffn_activation=dinc.FFN_ACTIVATION,
                max_len=dinc.MAXLEN,
                dnn_dropout=dinc.DROUPOUT_RATIO)

    model.compile(optimizer=dinc.OPTIMIZER,loss=dinc.LOSS_FUNCTION,metrics=dinc.METRIC)

    model.fit(
        train_X,
        train_y,
        epochs=dinc.EPOCHS,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y),
        batch_size=dinc.BATCH_SIZE,
    )