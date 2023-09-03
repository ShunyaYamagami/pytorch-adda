cd ~/lab/gda/da/pytorch-adda

. cmd.sh 0 0  -1 --parent Office31  --task true_domains    --tmux ADDA__GPU01
. cmd.sh 0 0  4_5  --parent OfficeHome  --task true_domains    --tmux ADDA__GPU02
. cmd.sh 1 0  0_1  --parent OfficeHome  --task true_domains    --tmux ADDA__GPU11
. cmd.sh 1 0  2_3  --parent OfficeHome  --task true_domains    --tmux ADDA__GPU12

. cmd.sh 1 0  -1 --parent Office31  --task simclr_bs512_ep300_g3_shfl    --tmux ADDA__GPU01
. cmd.sh 1 0  4_5  --parent OfficeHome  --task simclr_bs512_ep1000_g3_shfl    --tmux ADDA__GPU02
. cmd.sh 0 0  0_1  --parent OfficeHome  --task simclr_bs512_ep1000_g3_shfl    --tmux ADDA__GPU11
. cmd.sh 0 0  2_3  --parent OfficeHome  --task simclr_bs512_ep1000_g3_shfl    --tmux ADDA__GPU12

. cmd.sh 0 0  0_2_4  --parent OfficeHome  --task simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl/    --tmux ADDA__GPU11
. cmd.sh 1 0  1_3_5  --parent OfficeHome  --task simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl/    --tmux ADDA__GPU12

