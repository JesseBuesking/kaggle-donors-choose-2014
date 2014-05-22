

from datetime import datetime
from kaggle_donors_choose_2014.model_runner import ModelRunner


def save_stats(filename, stats):
    with open(filename, 'wt') as f:
        for line in stats:
            f.write('{}\n'.format(line))


def save_predictions(filename, predictions):
    with open(filename, 'wt') as f:
        f.write('projectid,is_exciting\n')
        predictions = list(predictions.itertuples())
        for prediction in predictions:
            f.write("%s,%s\n" % (
                prediction[1],
                prediction[2]
            ))


if __name__ == '__main__':
    mr = ModelRunner()

    model_name = 'naive-essay-length'
    mr.init(model_name)

    # train
    predictions = mr.train_model()
    sname = 'output/{}-{}.stats.txt'.format(
        datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        model_name
    )
    stats = mr.stats(predictions)
    print('')
    for line in stats:
        print(line)
    save_stats(sname, stats)

    # test
    final_predictions = mr.test_model()

    # save to file
    oname = 'output/{}-{}.csv'.format(
        datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        model_name
    )
    save_predictions(oname, final_predictions)
