
with open('../dimsum16.test_encoded.test', 'w', encoding='utf-8') as fw:
    with open('../dimsum16.test', 'r', encoding='utf-8') as fr:
        X, y_mwe, y_sst = [], [], []
        for i, line in enumerate(fr.readlines()):
            if line != '\n':
                num, word, lem, pos, mwe, parent, tab, supersenses, id_ref = line[:-1].split('\t')
                try:
                    fw.write(str(f'{num}\t{word}\t{lem}\t{pos}\t{mwe}\t{parent}\t{tab}\t{supersenses}\t{id_ref}\n'.encode('utf-8'), 'cp1252'))
                except (UnicodeDecodeError, UnicodeEncodeError):
                    fw.write(str(f'{num}\t---\t---\t{pos}\t{mwe}\t{parent}\t{tab}\t{supersenses}\t{id_ref}\n'.encode('utf-8'), 'cp1252'))
            else :
                fw.write('\n')

