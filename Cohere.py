# pip install cohere
import cohere
# API key (trial)
co = cohere.Client('VgR2hXk1OC9UOiTWFYE1rTodw1GkT7xYKI6MsLIS') 

prompt0 = 'dog'

response = co.generate(
  model='command-nightly',
  prompt='how to draw ' + prompt0,
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')

result = response.generations[0].text
print('Prediction: {}'.format(result))
print(' ')
print(result.strip('\n'))