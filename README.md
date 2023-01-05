# PPO-baselines-mario
SuperMarioBros Nes CNN using PPO from stable-baselines. Changes input shape from 240,256,3 to 14,14,4.
Uses some utility functions from Chrispresso @GitHub: https://github.com/Chrispresso/SuperMarioBros-AI.

Example of the trained NN completing the first level.

![PPOmariofinishinglevel](https://user-images.githubusercontent.com/97381129/210824973-8d004373-48d2-4e23-a10e-d9a9a8f5e1a3.gif)


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

The input is only 14,14,4 so a complex model is not needed, and even this is likely more complex than is needed.
