import time
import tensorflow as tf

@tf.function
def train_step(model, features, targets, loss_object, optimizer,
    train_loss, train_metric, output_transform=None):
    """
    Training step

    Parameters
    ----------
    model : tensorflow.Module

    features : tensorflow.Tensor

    targets : tensorflow.Tensor


    """
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(features, training=True)
        if output_transform:
            predictions = output_transform(predictions)
        loss = loss_object(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_metric(targets, predictions)

@tf.function
def test_step(model, features, targets, loss_object, test_loss, test_metric, output_transform=None):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    predictions = model(features, training=False)
    if output_transform:
        predictions = output_transform(predictions)
    t_loss = loss_object(targets, predictions)

    test_loss(t_loss)
    test_metric(targets, predictions)

def max_pooler(inputs):
    ksize = inputs.shape[1]
    return tf.squeeze(tf.nn.max_pool1d(inputs,ksize,1,"VALID"))


def train(model, optimizer, loss_object, train_loss, train_dataset, train_metric,
    val_loss, val_dataset, val_metric, output_transform=None, epochs=5):
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train RMSE':^9} " +
          f"| {'Val Loss':^10} | {'Val RMSE':^9} | {'Elapsed':^9}")
    print("-"*60)
    best_val_loss = None
    for epoch in range(epochs):
        t0_epoch = time.time()
        train_loss.reset_state()
        train_metric.reset_state()
        val_loss.reset_state()
        val_metric.reset_state()

        for train_features, train_targets in train_dataset():
            train_step(model=model, features=train_features, targets=train_targets,
            loss_object=loss_object, optimizer=optimizer, train_loss=train_loss,
            train_metric=train_metric, output_transform=output_transform)

        for val_features, val_targets in val_dataset():
            test_step(model, val_features, val_targets,
            loss_object, val_loss, val_metric)

        time_elapsed = time.time() - t0_epoch
        print(f"{epoch+1:^7} | {train_loss.result():^12.6f} | {train_metric.result():^9.2f}" +
              f" | {val_loss.result():^10.6f}" +
              f" | {val_metric.result():^9.2f} | {time_elapsed:^9.2f}")

        if best_val_loss is not None and val_loss.result() > best_val_loss:
            print("Stopping early: Val loss increased")
            break
        else:
            best_val_loss = val_loss.result()
    print(f"Training completed! Final validation RMSE: {val_metric.result():.2f}.")