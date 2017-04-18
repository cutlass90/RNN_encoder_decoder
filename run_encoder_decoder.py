from encoder_decoder import EncoderDecoder
import tools


encoder_decoder = EncoderDecoder(embedding_size=300, n_hidden_RNN=128,
    do_train=True)

# Training
data_provider = tools.DataProvider(path_to_csv='dataset/set_for_encoding.csv',
    path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
    # path_to_w2v='~/GoogleNews-vectors-negative300.bin',    
    test_size=0.15)

encoder_decoder.train_(data_loader=data_provider, keep_prob=1, weight_decay=0.005,
    learn_rate_start=0.005, learn_rate_end=0.0003, batch_size=64, n_iter=100000,
    save_model_every_n_iter=5000, path_to_model='models/siamese')


"""
#Evaluating COST
encoder_decoder.eval_cost(data_loader=data_provider, batch_size=512,
    path_to_model='models/siamese')
"""


#Prediction
data_provider = tools.DataProvider(path_to_csv='dataset/temp.csv',
    path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
    # path_to_w2v='~/GoogleNews-vectors-negative300.bin',    
    test_size=0)

encoder_decoder.predict(batch_size=512, data_loader=data_provider, path_to_save='recovered.txt',
    path_to_model='models/siamese')
