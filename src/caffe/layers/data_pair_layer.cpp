#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_pair_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataPairLayer<Dtype>::DataPairLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  //db_.reset(db::GetDB(param.data_param().backend()));
  db_.reset(db::GetDB("lmdb"));
  db_->Open(param.data_pair_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
DataPairLayer<Dtype>::~DataPairLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataPairLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_pair_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  DatumPair datumpair;
  datumpair.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape(4);
  top_shape[0] = 2;
  top_shape[1] = 3;
  top_shape[2] = 112;
  top_shape[3] = 112;
  this->transformed_data_.Reshape(top_shape);
  std::cout << this->transformed_data_.shape(1);
  std::cout << top_shape[2] << std::endl;
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size*2;
  top[0]->Reshape(top_shape);
  std::cout << top_shape.data() << std::endl;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}

template <typename Dtype>
bool DataPairLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void DataPairLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void DataPairLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    std::cout << "Yes" << std::endl;
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_pair_param().batch_size();
  std::cout << "ff" << std::endl;
  DatumPair datumpair;
  std::cout << batch_size << std::endl;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      std::cout << "fff1" << std::endl;
    timer.Start();
    while (Skip()) {
      Next();
    }
    datumpair.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape(4);
      top_shape[0] = 2;
      top_shape[1] = 3;
      top_shape[2] = 112;
      top_shape[3] = 112;
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size * 2;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id * 2);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datumpair, &(this->transformed_data_));

    // Copy label.
    if (this->output_labels_) {
      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datumpair.label();
      std::cout << "label: " << datumpair.label() << std::endl;
    }
    trans_time += timer.MicroSeconds();
    Next();
    std::cout << "fff" << std::endl;
  }
//exit(0);
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataPairLayer);
REGISTER_LAYER_CLASS(DataPair);

}  // namespace caffe
