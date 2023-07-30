function process_args {
    declare -A args
    # 無名引数数
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3  # -1の時, 前dsetを実行
    shift 3  # 無名引数数

    # 残りの名前付き引数を解析
    local parent="OfficeHome"
    if [ $parent = 'Office31' ]; then
        local task=(
            # "original_uda"
            # "true_domains"
            # "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            # "simclr_bs512_ep300_g3_shfl"
            "simple_bs512_ep300_g3_AE_outd64_shfl"
            "contrastive_rpl_dim512_wght0.6_AE_bs256_ep300_outd64_g3"
        )
    elif [ $parent = 'OfficeHome' ]; then
        local task=(
            # "original_uda"
            "true_domains"
            "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep300_g3_shfl"
        )
    fi
    
    local params=$(getopt -n "$0" -o p:t: -l parent:,task: -- "$@")
    eval set -- "$params"

    while true; do
        case "$1" in
            -p|--parent)
                parent="$2"
                shift 2
                ;;
            # -t|--task)
            #     task="$2"
            #     shift 2
            #     ;;
            --)
                shift
                break
                ;;
            *)
                echo "不明な引数: $1" >&2
                return 1
                ;;
        esac
    done
    echo "gpu_i: $gpu_i"
    echo "exec_num: $exec_num"
    echo "dset_num: $dset_num"
    echo "parent: $parent"
    echo "task: $task"
    echo -e ''  # (今は使っていないが)改行文字は echo コマンドに -e オプションを付けて実行した場合にのみ機能する.
    
    ###################################################
    ##### データセット設定
    if [ $parent = 'Office31' ]; then
        dsetlist=("amazon_dslr" "webcam_amazon" "dslr_webcam" )
    elif [ $parent = 'OfficeHome' ]; then
        dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
    elif [ $parent = 'DomainNet' ]; then
        dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
    else
        echo "不明なデータセット: $parent" >&2
        return 1
    fi
    
    ###################################################
    COMMAND="conda deactivate && conda deactivate "
    COMMAND+=" && conda activate tvt "
    for tsk in "${task[@]}"; do
        if [ $dset_num -eq -1 ]; then
            for dset in "${dsetlist[@]}"; do
                COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python  main.py  --parent $parent --dset $dset  --task $tsk"
            done
        else
            dset=${dsetlist[$dset_num]}
            COMMAND+=" &&  CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python  main.py  --parent $parent --dset $dset  --task $tsk"
        fi
    done

    ###################################################
    ###### 実行. 
    echo $COMMAND
    echo ''
    eval $COMMAND
}

####################################################
########## Verify the number of arguments ##########
# 最初の3つの引数をチェック
if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"