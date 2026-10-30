import jiwer

try:
    ground_truth = "hello world"
    hypothesis = "hello duck"
    
    print(f"jiwer file: {jiwer.__file__}")
    print("Attempting to call jiwer.process...")
    
    output = jiwer.process(ground_truth, hypothesis)
    
    print("jiwer.process call was successful!")
    print(output.wer)
    print(output.alignments)

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Type of 'jiwer': {type(jiwer)}")
    print(f"dir(jiwer): {dir(jiwer)}")
    try:
        print(f"Type of 'jiwer.process': {type(jiwer.process)}")
        print(f"dir(jiwer.process): {dir(jiwer.process)}")
    except Exception as e2:
        print(f"Could not inspect jiwer.process: {e2}")

