import tempfile
import unittest
from pathlib import Path

import gensim
import numpy as np
import tensorflow as tf


class NotebookDependencyCompatibilityTests(unittest.TestCase):
    def test_text_model_training_and_safe_roundtrip(self):
        tf.keras.utils.set_random_seed(7)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,), dtype="int32"),
            tf.keras.layers.Embedding(8, 4),
            tf.keras.layers.LSTM(4),
            tf.keras.layers.Dense(2),
        ])
        model.compile(optimizer="adam", loss="mse")
        inputs = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int32)
        loss = model.train_on_batch(inputs, np.zeros((2, 2), dtype=np.float32))
        self.assertTrue(np.isfinite(loss))
        expected = model(inputs).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path, safe_mode=True)
            np.testing.assert_allclose(restored(inputs).numpy(), expected, atol=1e-6)

    def test_word_embeddings_keep_gensim_and_scipy_compatible(self):
        model = gensim.models.Word2Vec(
            sentences=[["hello", "world"], ["hello", "tensorflow"]],
            vector_size=4,
            min_count=1,
            workers=1,
            seed=7,
            epochs=1,
        )
        self.assertEqual(model.wv["hello"].shape, (4,))
        self.assertTrue(np.isfinite(model.wv["hello"]).all())


if __name__ == "__main__":
    unittest.main()
