import tensorflow_datasets as tfds
import tensorflow as tf

model = tf.keras.models.load_model('dr_deploy/model/my_model.h5')

model.summary()

encoder = tfds.features.text.SubwordTextEncoder.load_from_file('dr_deploy/model/encoder.dat')

mytext = "Not very good at all. Terrible script and acting. Difficult to understand and very, very bleak"
myrec = encoder.encode(mytext)
rez = model.predict([myrec])

print("Scored record: [", mytext, "] result: ", rez)



