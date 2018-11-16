import tensorflow as tf

def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  m1 = tf.shape(mat1)[0]
  n1 = tf.size(mat1) // m1
  m2 = tf.shape(mat2)[0]
  n2 = tf.size(mat2) // m2
  mat1_rsh =tf.reshape(mat1, [m1, 1, n1, 1])
  mat2_rsh =tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2]),[-1])