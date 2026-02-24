import { useDropzone } from 'react-dropzone'

const ACCEPTED = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'application/dicom': ['.dcm', '.dicom'],
  'application/octet-stream': ['.dcm', '.dicom'],
}

export default function DicomUploader({ onFile, file }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: ACCEPTED,
    multiple: false,
    onDrop: (accepted) => {
      if (accepted.length > 0) onFile(accepted[0])
    },
  })

  return (
    <div
      {...getRootProps()}
      className={`relative flex flex-col items-center justify-center gap-3
        border-2 border-dashed rounded-xl p-10 cursor-pointer transition-colors duration-150
        ${isDragActive
          ? 'border-green-400 bg-green-950/30'
          : 'border-gray-700 hover:border-gray-500 bg-gray-900'
        }`}
    >
      <input {...getInputProps()} />
      <div className="text-4xl select-none">📂</div>
      {file ? (
        <div className="text-center">
          <p className="text-green-400 font-semibold">{file.name}</p>
          <p className="text-gray-500 text-sm mt-1">
            {(file.size / 1024).toFixed(1)} KB — değiştirmek için tıklayın
          </p>
        </div>
      ) : (
        <div className="text-center">
          <p className="text-gray-300 font-medium">
            {isDragActive ? 'Dosyayı bırakın…' : 'DICOM / PNG / JPEG dosyasını sürükleyin'}
          </p>
          <p className="text-gray-500 text-sm mt-1">veya tıklayarak seçin</p>
          <p className="text-gray-600 text-xs mt-2">.dcm .dicom .png .jpg .jpeg</p>
        </div>
      )}
    </div>
  )
}
