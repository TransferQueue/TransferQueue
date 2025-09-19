from tensordict import TensorDict

from transfer_queue.data_system import StorageUnitData

if __name__ == '__main__':
    data_to_put = TensorDict(
        a = [
            "123", "123", None, "",
        ],
        b = [
            [3, 2, 1], [7, 6, 5, 4], [12, 11, 10, 9, 8], [14, 13],
        ]
    )
    sd = StorageUnitData(storage_size=10)

    sd.put_data(
        field_data=data_to_put,
        local_indexes=[1, 2, 3, 4]
    )

    print(sd.field_data)

    res = sd.get_data(fields=["a", "b"], local_indexes=[1, 2, 3, 4])

    print(res["a"])
    print(res["b"])
    print(res["a"][0])
    print(res["b"][2])