# This is the function to save the results.
# the inputs are: Method, modelname, epochs, batch_size, accuracy, f1score_macro, number of kerns, and filter size
# pylint: disable-all
def saveResultsCSV( aug_method, aug_factor, modelname, accuracy):
    import os.path
    from pathlib import Path
    from datetime import datetime

    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/har.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write('Finished on;aug_method;aug_factor; modelname; accuracy\n')
        f.close()

    with open(fileString, "a") as f:
        f.write('{};{};{};{}\n'.format(now, aug_method, modelname, aug_factor, accuracy))
        f.close()
