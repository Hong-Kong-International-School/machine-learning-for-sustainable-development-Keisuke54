import cohere
import time

# API key 
co = cohere.Client('VgR2hXk1OC9UOiTWFYE1rTodw1GkT7xYKI6MsLIS') 

qList = ['Tell me about landscape sketching/drawing.', 'How do you start landscape sketching/drawing?', 'What are some techniques for landsape sketching/drawing?', 'What are the criteria for landscape sketching/drawing?','How do you evaluate landscape sketching/drawing?']
conversation = []
A = ''
B = ''
tempConv = ''

for i in range(2):
    init = qList[i%5]
    A = init
    tempConv = 'A: ' + A 
    for i in range(5):
        if i % 2 == 0:
            response = co.generate(
                model='command-nightly',
                prompt=A,
                max_tokens=300,
                temperature=0.9,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            B = response.generations[0].text
            tempConv = tempConv + '\n' + 'B: ' + B 
        else:
            response = co.generate(
                model='command-nightly',
                prompt=B,
                max_tokens=300,
                temperature=0.9,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            A = response.generations[0].text
            tempConv = tempConv + '\n' + 'A: ' + A 
    
    conversation.append(tempConv)
    time.sleep(60)

with open("conversation.py", 'w') as conv:
    conv.write("conversation = [\n")    
    for line in conversation:
        conv.write('\'\'\''+ line + '\'\'\',\n')
    conv.write("]")
    
print('end')