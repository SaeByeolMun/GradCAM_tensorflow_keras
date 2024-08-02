import matplotlib.cm as cm
import tensorflow as tf
import numpy as np

class GradCAM():
    def __init__(self, model, last_conv_layer_name=''):
       
        self.model = model
        
        model_1, model_2 = self.model_12(model, last_conv_layer_name)
        self.model_1 = model_1
        self.model_2 = model_2
    
    def preprocessing_img(self, img):
        preprocessed_img = np.expand_dims(img, 0) / 255.0
        return preprocessed_img
    
    def model_12(self, model, last_conv_layer_name):
        '''
        모델을 두 개로 분할하는 메서드
        model = 학습된 tensorflow 모델
        last_conv_layer_name = 마지막 convolution layer의 레이어명 (model.summary()로 확인하거나 model.layers[1].name로 가져올 것)
        '''

        if last_conv_layer_name=='':
            last_conv_layer_name = model.layers[-3].name
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # 첫 번째 단계, 입력 이미지를 마지막 컨볼루션 층의 activation으로 매핑하는 모델을 만듦.
        model_1 = tf.keras.Model(model.inputs, last_conv_layer.output)
        
        # 두 번째 단계, 마지막 컨볼루션 층의 activation을 최종 클래스 예측으로 매핑하는 모델을 만듦.
        model_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x_2 = model_input
        x_2 = model.get_layer(model.layers[-2].name)(x_2)
        x_2 = model.get_layer(model.layers[-1].name)(x_2)
        model_2=tf.keras.Model(model_input, x_2)
        
        return model_1, model_2
        
    def make_gradcam_heatmap(self, img):
        img = self.preprocessing_img(img)
        with tf.GradientTape() as tape:
            # 마지막 컨볼루션 층의 activation을 계산하고 tape가 이를 확인함.
            last_conv_layer_output = self.model_1(img)
            tape.watch(last_conv_layer_output)
            # 클래스 예측을 계산.
            preds = self.model_2(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
    
        # 마지막 컨볼루션 층의 출력 특징 맵에 대한 최상위 예측 클래스의 gradient.
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        
        # 이 벡터의 각 항목은 특정 특징 맵 채널에 대한 그라디언트의 평균 강도.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
        # 마지막 컨볼루션 층 출력의 각 채널에 대해
        # "이 채널이 최상위 예측 클래스에 얼마나 중요한지"를 곱함.
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
        # 결과적인 특징 맵의 채널별 평균이 클래스 활성화 히트맵.
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # 시각화를 위해 히트맵을 0과 1 사이로 정규화.
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap
    
    def return_color_heatmap(self, img): 
        '''
        img =  model predict를 진행할 input image
        '''
        # GradCAM 히트맵을 생성.
        heatmap = self.make_gradcam_heatmap(img)
        # 히트맵을 [0, 255] 범위의 uint8 타입으로 변환.
        heatmap =np.uint8(255*heatmap) 
        
        # 'jet' 컬러맵을 가져옴.
        jet = cm.get_cmap("jet")
        # 'jet' 컬러맵의 색상을 [0, 255] 범위의 배열로 변환.
        color = jet(np.arange(256))[:,:3]
        # 히트맵의 값을 'jet' 컬러로 변환.
        color_heatmap = color[heatmap]
         
        # 컬러 히트맵을 PIL 이미지로 변환.
        color_heatmap = tf.keras.preprocessing.image.array_to_img(color_heatmap)
        # 원본 이미지 크기로 리사이즈.
        color_heatmap = color_heatmap.resize((img.shape[1], img.shape[0]))
        # 다시 배열로 변환.
        color_heatmap = tf.keras.preprocessing.image.img_to_array(color_heatmap)
         
        # 컬러 히트맵을 원본 이미지 위에 덧씌움.
        overlay_img= color_heatmap * 0.5 + img
        # 최종 이미지를 PIL 이미지로 변환.
        overlay_img = tf.keras.preprocessing.image.array_to_img(overlay_img)
        return overlay_img

if __name__ == "__main__": 
    ...
