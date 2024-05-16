This was a test to see whether it is neccesary to explicitly call "accelerate" when training a LLM using the Huggingface trainer API.
Result was that it wasn't necessary.
The rest consisted of running test_classify.py while varrying the number of GPU's availible in run_test.sh from 1 to 4. This was done for calling python3 and accelerate 
