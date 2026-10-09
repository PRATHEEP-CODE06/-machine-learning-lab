# Implementation of FIND-S Algorithm

# Training dataset: [Sky, AirTemp, Humidity, Wind, Water, Forecast, EnjoySport]
dataset = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

def find_s(data):
    num_attributes = len(data[0]) - 1
    # Initialize most specific hypothesis
    hypothesis = ['0'] * num_attributes
    print("Initial Hypothesis:")
    print(hypothesis)
    print("-" * 50)
    
    # Process each training instance
    for i, instance in enumerate(data):
        target = instance[-1]
        features = instance[:-1]
        
        if target.lower() == 'yes':
            print(f"Instance {i + 1} is Positive: {features}")
            for j in range(num_attributes):
                if hypothesis[j] == '0':
                    hypothesis[j] = features[j]
                elif hypothesis[j] != features[j]:
                    hypothesis[j] = '?'
            print(f"Hypothesis after Instance {i + 1}: {hypothesis}")
        else:
            print(f"Instance {i + 1} is Negative (Ignored): {features}")
        print("-" * 50)
        
    return hypothesis

# Execute FIND-S
print("Given Training Dataset:")
for row in dataset:
    print(row)
print("=" * 50)

final_hypothesis = find_s(dataset)
print("\nFinal Maximally Specific Hypothesis:")
print(final_hypothesis)