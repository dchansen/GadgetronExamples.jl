module ReconBuckets 
using FFTW
using Gadgetron
using Gadgetron.MRD
using Gadgetron.Types:AcquisitionBucket,ReconData 
using Gadgetron.Stream
using Gadgetron.Default 
using PartialFunctions
using LinearAlgebra
using Setfield

fft_scaling(xsize,dims=[1]) = sqrt(reduce((*),map(d->xsize[d],dims)))

ortho_fft(x,dims=[1]) = fft(x,dims) ./ Float32(fft_scaling(size(x),dims))

ortho_ifft(x,dims=[1]) = bfft(x,dims) ./ Float32(fft_scaling(size(x),dims))

cfft(x, dims=[1]) = ifftshift(ortho_fft(fftshift(x,dims),dims),dims)


cifft(x, dims=[1]) = fftshift(ortho_ifft(ifftshift(x,dims),dims),dims)



function make_kspace(header::MRDHeader)

	matrix_size = header.encoding[1].encodedSpace.matrixSize
	
	function inner(bucket::AcquisitionBucket)
        acqs = bucket.data
		number_of_samples, number_of_channels = size(acqs[1].data)
		buffer = zeros(ComplexF32,number_of_samples,matrix_size.y,matrix_size.z,number_of_channels)
		for acq in acqs 
			buffer[:,acq.header.idx.kspace_encode_step_1+1,acq.header.idx.kspace_encode_step_2+1,:] = acq.data
		end
		return [(acqs[1].header,buffer)]
		
	end

    function inner(buffer::ReconData)
        data= buffer[1].data.data 
        dims = size(data )
        data_view = reshape(data,dims[1],dims[2],dims[3],dims[4],:)
        return (( buffer[1].data.header[1], data) for data in eachslice(data_view,dims=5))
    end

	return MapCat(inner)
end

function reconstruct_image(header::MRDHeader)

    field_of_view = header.encoding[1].reconSpace.fieldOfView_mm
    fov = (field_of_view.x,field_of_view.y,field_of_view.z)

	recon(x) = cifft(x,[1,2,3]) 
    coil_combine(x) = sqrt.(sum(abs2,x,dims=4))

	inner(index, reference::AcquisitionHeader, buffer) =   Image(ImageHeader(reference; image_index=index, image_type = MRD.ImageType.magnitude, field_of_view=fov),  coil_combine(recon(buffer)))
    inner_splat(args ) = inner(args[1],args[2][1],args[2][2])
	return Map(inner_splat)
end

function reconstruct_buckets(connection)
    start = time()
    header = connection.header
    connection |>  
    make_kspace(header) |>
    Enumerate() |>
    reconstruct_image(header) .|>
    push! $ connection

    tot_time = time()-start
    @info "Total recon time $tot_time"
end

Base.precompile(Tuple{typeof(reconstruct_buckets),Gadgetron.MRDChannel})   # time: 0.5008968

end
