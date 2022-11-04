from util.Encoder import Encoder
from util.Parser import Parser

if __name__ == "__main__":
    parser = Parser()
    encoder = Encoder()

    review_id_keys, tf_keys, pn_keys = parser.parse_keys()
    tf_keys = encoder.encode_label(tf_keys, label_p='true', label_n='fake')
    pn_keys = encoder.encode_label(pn_keys, label_p='pos', label_n='neg')

    review_id_preds, tf_preds, pn_preds = parser.parse_output()
    tf_preds = encoder.encode_label(tf_preds, label_p='true', label_n='fake')
    pn_preds = encoder.encode_label(pn_preds, label_p='pos', label_n='neg')

    print('True/False')
    num_acc = 0
    total = len(tf_keys)
    for i in range(len(tf_keys)):
        if tf_keys[i] == tf_preds[i]:
            num_acc += 1

    print(f'Acc: {num_acc}/{total}')

    print('Pos/Neg')
    num_acc = 0
    total = len(pn_keys)
    for i in range(len(pn_keys)):
        if pn_keys[i] == pn_preds[i]:
            num_acc += 1

    print(f'Acc: {num_acc}/{total}')