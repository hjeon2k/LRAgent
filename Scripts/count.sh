TYPE=("hotpotqa" "scienceqa")
QNUM=(100 120)  # per level question number
ITER=(20 20)  # per level iter number

DIR="$1"
IDX="$2"

cd "$DIR" || exit 1

total_q=$(( ${QNUM[$IDX]} * ${ITER[$IDX]} ))
total_correct=$(grep -rIwo "CORRECT" .   | wc -l | awk '{print $1}')
total_incorrect=$(grep -rIwo "INCORRECT" . | wc -l | awk '{print $1}')

awk -v C="$total_correct" -v I="$total_incorrect" -v T="$total_q" -v TT="${TYPE[$IDX]}" '
BEGIN{
  acc   = C / T * 100
  inacc = I / T * 100
  O = T - C - I
  if (O < 0) O = 0
  printf "Task type: %s\n", TT
  printf "Total questions: %d\n", T
  printf "Total CORRECT: %d (acc=%.2f)\n", C, acc
  printf "Total INCORRECT: %d (inacc=%.2f)\n", I, inacc
  printf "OOD num: %d\n", O
}'

grep -rIwo "CORRECT" . | cut -d: -f1 | sort | uniq -c | awk -v Q="${QNUM[$IDX]}" -v IT="${ITER[$IDX]}" '
{
  a = $1 / Q;
  acc[NR] = a;
  sum += a;
}
END{
  mean = sum / IT;
  for (i = 1; i <= IT; i++) ss += (acc[i] - mean) * (acc[i] - mean);
  sem  = sqrt(ss / IT / (IT-1));
  printf "SEM: %.2f\n", sem * 100;
}'
