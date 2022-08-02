import jax
import jax.numpy as np
import optax

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, num_classes: int) -> float:
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    loss *= mask
    return np.sum(loss)
