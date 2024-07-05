# Following are the Emotions in the RAVDESS dataset & ASVP-ESD dataset, the datasets we using for this model training.
# But we observe only the emotions in the observed_emotions list below.
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised',
  '09':'',
  '10':'',
  '11':'',
  '12':'',
    '13': 'happy',  
    '23': 'gaggle',
    '33': 'happy',
    '14': 'sad', 
    '24': 'sigh',
    '34': 'sniffle',
    '44': 'suffering',
    '16': 'fearful',  
    '36': 'panic',
    '15': 'angry',  
    '25': 'frustration',
    '35': 'angry',
    '0': 'grunt',
    '18': 'surprised', 
    '28': 'amazed',
    '38': 'astonishment',
    '48': 'disgust',
    '17': 'disgust',
    '27': 'rejection'
}

#### But!!!
# Emotions to observe : only following
observed_emotions=['happy', 'sad','angry', 'fearful']   
