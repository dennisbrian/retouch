"""Delivery writer: copy audio, or losslessly encode PCM16 as ALAC for MP4."""
from fractions import Fraction
from .media import VideoWriter
import av


class DeliveryWriter(VideoWriter):
    """Streaming audio strategy with abort-aware cleanup.

    AAC/ALAC and compatible container audio are copied unchanged. PCM16 MOV
    input becomes ALAC in MP4; decoded samples and timing are verified by the
    job. MKV retains stream copy. On failure/cancellation, do not drain the
    remainder of a long source audio stream into a discarded partial video.
    """
    def __init__(self,output,source,*,lossless=False,crf=18):
        self._pcm_container=None
        self._pcm_pending=None
        self._pcm_flushed=False
        with av.open(str(source)) as container:
            codec=container.streams.audio[0].codec_context.name if container.streams.audio else None
        transcode=not lossless and codec in ('pcm_s16le','pcm_s16be')
        if not lossless and codec and codec.startswith('pcm_') and not transcode:
            raise ValueError('MP4 PCM conversion currently supports PCM16; use lossless MKV for other PCM formats')
        super().__init__(output,source,lossless=lossless,crf=crf,audio=not transcode)
        self.audio_mode='pcm16_to_alac_lossless' if transcode else 'copy' if codec else 'silent'
        if transcode:
            try:
                self._pcm_container=av.open(str(source))
                audio=self._pcm_container.streams.audio[0]
                rate=audio.codec_context.sample_rate
                layout=audio.codec_context.layout.name
                self._pcm_frames=iter(self._pcm_container.decode(audio))
                self._pcm_encoder=self._out.add_stream('alac',rate=rate)
                self._pcm_encoder.layout=layout
                self._pcm_encoder.format='s16p'
                self._pcm_encoder.time_base=Fraction(1,rate)
                self._pcm_encoder.codec_context.time_base=Fraction(1,rate)
                self._pcm_resampler=av.AudioResampler(format='s16p',layout=layout,rate=rate)
            except BaseException:
                self.abort()
                raise

    def _mux_audio_until(self,seconds):
        if self._pcm_container is None:
            return super()._mux_audio_until(seconds)
        if self._pcm_flushed:return
        while True:
            frame=self._pcm_pending if self._pcm_pending is not None else next(self._pcm_frames,None)
            self._pcm_pending=None
            if frame is None:break
            if frame.pts is None:raise ValueError('PCM audio has no presentation timestamp')
            if seconds is not None and float(frame.pts*frame.time_base)>seconds:
                self._pcm_pending=frame
                return
            for converted in self._pcm_resampler.resample(frame):
                for packet in self._pcm_encoder.encode(converted):self._out.mux(packet)
        if seconds is None:
            for converted in self._pcm_resampler.resample(None):
                for packet in self._pcm_encoder.encode(converted):self._out.mux(packet)
            for packet in self._pcm_encoder.encode():self._out.mux(packet)
            self._pcm_flushed=True

    def abort(self):
        """Close owned containers without flushing the whole source audio."""
        try:self._out.close()
        finally:
            self._source.close()
            if self._pcm_container is not None:self._pcm_container.close()

    def close(self):
        """Finalize video/audio and close the extra PCM decoder."""
        try:super().close()
        finally:
            if self._pcm_container is not None:self._pcm_container.close()

    def __exit__(self,exc_type,*args):
        if exc_type is None:self.close()
        else:self.abort()
