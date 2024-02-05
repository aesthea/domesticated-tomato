//2023-08-15 Original will cause infinite loop of websocket send. This version use promise to prevent another event trigger.
function fsdf(id, output_image){
	return new Promise((resolve) => {
		const input = document.getElementById(id)
		const maxsize = 640;
		var file = input.files[0];
		var original_file_size = input.files[0].size;
		read = new FileReader();
		read.onloadend = function(){
			//console.log(read.result);
			var raw_value = read.result;
			var canvas = document.createElement("canvas");
			var ctx = canvas.getContext("2d");
			var image = new Image();
			image.onload = function() {
				//console.log("IMAGE ORIGINAL SIZE", image.width, image.height);
				create_msg("IMAGE ORIGINAL SIZE " + image.width + " " + image.height);
				var ratio = image.width / image.height;
				if(ratio > 1){
					if(image.width > maxsize){
						canvas.width = maxsize;
						canvas.height = parseInt(maxsize / ratio);
					}else{
						canvas.width = image.width;
						canvas.height = image.height;
					}
				}else{
					if(image.height > maxsize){
						canvas.width = parseInt(maxsize * ratio);
						canvas.height = maxsize;
					}else{
						canvas.width = image.width;
						canvas.height = image.height;
					}
				}
				ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
				//console.log("IMAGE NEW SIZE", canvas.width, canvas.height);
				create_msg("IMAGE NEW SIZE " + canvas.width + " " + canvas.height);
				
				//https://stackoverflow.com/questions/27251953/how-to-create-file-object-from-blob
				//var dataUrl = canvas.toDataURL(file.type);
				//console.log("FILE TYPE", file.type);
				var dataUrl = canvas.toDataURL("image/jpeg");
				var bytes = atob(dataUrl.split(',')[1]);
				var mime = dataUrl.split(',')[0].split(':')[1].split(';')[0];
				mime = "image/jpeg";
				var max = bytes.length;
				var ia = new Uint8Array(max);
				for (var i = 0; i < max; i++) {
					ia[i] = bytes.charCodeAt(i);
				}
				var newImageFileFromCanvas = new File([ia], file.name, { type: mime });
				
				//console.log("SIZE COMPARE old:", original_file_size, "new:", newImageFileFromCanvas.size);
				create_msg("SIZE COMPARE old: " + original_file_size + " new: " + newImageFileFromCanvas.size);
				
				var dt = null;
				var image_dataurl = null;
				
				if(original_file_size > newImageFileFromCanvas.size){
					image_dataurl = dataUrl;
					//console.log("IMAGE IS REPLACED");
				}else{
					image_dataurl = image.src;
					//console.log("SKIP, OLD FILE SMALLER");
				}
				
				create_msg(image_dataurl);
				
				try{
					if(original_file_size > newImageFileFromCanvas.size){
						dt = new DataTransfer();
						create_msg("new DataTransfer() constructor");
						dt.items.add(newImageFileFromCanvas);
						create_msg("dt.items.add(newImageFileFromCanvas);");
						input.files = dt.files;
						create_msg("input.files = dt.files;");
					}else{
						create_msg("NO CHANGE SIZE");
					}
				}catch(e){
					create_msg(e);
				}
				
				document.getElementById(output_image).src = image_dataurl;
				
				const custom_event = new CustomEvent('RESIZED', {detail : {id : id, dt : new Date}});
				custom_event.initEvent('RESIZED', true, true);
				window.dispatchEvent(custom_event);
				resolve(id);
			}
			image.src = raw_value;
		}	
		read.readAsDataURL(file);
	})
}

async function filesize_down(id, output_image){
  const result = await fsdf(id, output_image);
  return result;
}

function create_msg(message){
	var root = document.getElementById("message");
	if(!root){
		return false;
	}
	var p = document.createElement('p');
	p.innerHTML = message;
	root.appendChild(p);
}

//https://stackoverflow.com/questions/35940290/how-to-convert-base64-string-to-javascript-file-object-like-as-from-file-input-f
function dataURLtoFile(dataurl, filename) {
	var arr = dataurl.split(','),
		mime = arr[0].match(/:(.*?);/)[1],
		bstr = atob(arr[1]), 
		n = bstr.length, 
		u8arr = new Uint8Array(n);
	while(n--){
		u8arr[n] = bstr.charCodeAt(n);
	}
	return new File([u8arr], filename, {type:mime});
}