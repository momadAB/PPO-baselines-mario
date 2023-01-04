# PPO-baselines-mario
SuperMarioBros Nes CNN using PPO from stable-baselines.
Uses some utility functions from Chrispresso @GitHub.

Check requirements.txt.

Uses (probably requires) Python3.7

MUST change lines in stable-baselines policies.py as follows:

lines 25-27 changed from:

    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    
to 

    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    
    layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    
Can change the network architecture to something else, but make sure that there are enough dimensions for whatever architecture you choose

Later changed to:

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = conv_to_fc(layer_3)
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
