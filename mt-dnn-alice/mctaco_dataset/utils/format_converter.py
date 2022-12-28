


def convert(in_file, out_file, id_count):
    f_in = open(in_file)
    f_out = open(out_file, "w")

    for line in f_in:
        data = line.strip().split("\t")

        f_out.write(str(id_count) +"\t"+ data[3]+"\t"+  data[0]+" "+data[1]+"\t"+ data[2])
        f_out.write("\n")
        id_count += 1

    f_out.flush()
    f_out.close()

    return id_count



id_count = 0

in_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/dev_3783.tsv"
out_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/canonical_data/MNLI/mnli_train.tsv"
id_count = convert(in_file, out_file, id_count)

in_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/dev_3783.tsv"
out_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/canonical_data/MNLI/mnli_dev.tsv"
id_count = convert(in_file, out_file, id_count)

in_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/test_9442.tsv"
out_file = "/Users/lisk/PycharmProjects/mt-dnn-vat-vat_update/data_mc_taco/canonical_data/MNLI/mnli_test.tsv"
convert(in_file, out_file, id_count)


