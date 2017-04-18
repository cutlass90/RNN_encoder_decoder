import tensorflow as tf

batch_size = 4
n_hidden = 5
max_step = 3
embeidding_size = 6

sess = tf.InteractiveSession()

en_state = tf.ones([batch_size, n_hidden])

decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(en_state)
dec_cell = tf.contrib.rnn.GRUCell(n_hidden)
inputs = tf.ones([batch_size, max_step, embeidding_size])
seq_length = [1,2,3,2]

outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
    cell=dec_cell,
    decoder_fn=decoder_fn,
    inputs=inputs,
    sequence_length=seq_length)


sess.run(tf.global_variables_initializer())

outputs_, final_state_ = sess.run([outputs, final_state])