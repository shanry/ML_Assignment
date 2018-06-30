把数据文件夹和LR_main放到同一目录下。
如下程序中每一行代表一个实验组设置，注释掉对应的行即可运行。
if __name__ == '__main__':
    # main(0, 0)      # raw
    main(5, 0)          # oversample
    # main(0, 0.4)    # move-threshold
    
例如运行第二行 main(5, 0) ,可获得输出：

test 538 examples label 1 acuu 0.9572490706319703
test 538 examples label 2 acuu 0.983271375464684
test 538 examples label 3 acuu 0.9944237918215614
test 538 examples label 4 acuu 0.9981412639405205
test 538 examples label 5 acuu 0.966542750929368
test acuu 0.9646840148698885
precision: [0.97535934 0.91666667 1.         0.8        0.625     ]
recall: [0.98547718 0.89189189 0.66666667 1.         0.41666667]