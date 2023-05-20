import cohere
import time

# API key 
co = cohere.Client('VgR2hXk1OC9UOiTWFYE1rTodw1GkT7xYKI6MsLIS') 

qList = ['What kind of concepts do I need to ask for AP Physics C?', 
         'Explain the concept of a normal force in physics.', 
         'What factors change the period of a simple harmonic motion pendulum?', 
         'How do you find the direction of induced current in a magnetic field?', 
         'How can you solve for the charge on a capacitor in a transient state by using differential equations?', 
         'What does increasing resistance in a RC circuit do?', 
         'What are the defining aspects of Simple Harmonic Motion?', 
         'How do you know which direction friction is for torque questions?', 
         'Whats the difference between Gausses law for electricity and Gausses law for magnetism?', 
         'conditions for using Gausses law', 'formulas related to torque', 
         'What is the general equation for Gausses law?', 
         'How can I find electric potential at a point?', 
         'How can I solve for a capacitance?', 
         'direciton of magnetic force',
         'How can I find the electric field for a uniform and insulating sphere?']

# tracking number of generations (max 5 per min)
i = 0

prompts = []
prompts = prompts + qList

# amplifying prompts 
amp = 7
for num in range(1):
    for q in qList:
        response = co.generate(
            model='command-nightly',
            prompt='generate ' + str(amp) + ' similar questions to '+ q,
            max_tokens=300,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
        qAmp = list(response.generations[0].text.split('\n'))
        prompts = prompts + qAmp
        i = i + 1
        if i%5 == 4:
            time.sleep(60)


prompt2 = [x for x in prompts if x != ""]
print(len(prompt2))
print(prompt2)

conversation = []
tempConv = ''
# creating conversation 
for p in prompt2:
    response = co.generate(
            model='command-nightly',
            prompt=p,
            max_tokens=300,
            temperature=0.9,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')
    refined = []
    bout = response.generations[0].text[2:]
    if bout[0:2] != '\n':
        bout = response.generations[0].text[1:]
    tempConv = 'Question: ' + p + '\n' + 'Answer: ' + bout
    conversation.append(tempConv)
    i = i + 1
    if i%5 ==4:
        time.sleep(60)

with open("conversation.py", 'w') as conv:
    conv.write("conversation = [\n")    
    for line in conversation:
        conv.write('\'\'\''+ line + '\n' + '\'\'\',\n' + '    ')
    conv.write("]")

print('end')