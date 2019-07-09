cd /home/rui/work/caffe
./build/tools/caffe train \
--solver="models/face_recognition/recognition/solver.prototxt" \
--weights="models/face_recognition/recognition/face_recognition_recognition_iter_31511.caffemodel" \
--gpu 0 2>&1 | tee jobs/face_recognition/recognition/face_recognition_recognition.log
