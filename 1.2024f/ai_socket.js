window.ANCHOR_SZ = 5;
window.INPUT_SZ = 128;
window.NMS_IOU = 0.01;
window.FSDN_size = 600;
window.IMAGE_FORMAT = "jpg";
window.SEGMENT_MIN_RATIO = 0.50;
window.ALLOWABLE_TAGS = new Array("ok", "ng");

SOCKLIST = new Array('ws://192.168.177.102:8860', 'ws://192.168.160.76:8790');

console.log("script ver 2024.11.21.15.47")
function socket_init(sockname){
	//console.log("socket_init", window.socket_retries);
	if(!window.hasOwnProperty("socket_list")){
		window.socket_list = new Object;
	}
	let sock_list;
	if(sockname){
		sock_list = new Array(sockname);
	}else{
		sock_list = SOCKLIST;
	}
	for(i in sock_list){
		let sockname = sock_list[i];
		window.socket_list[sockname] = new WebSocket(sockname);
		window.socket_list[sockname].onclose = async (event) => {
			console.error(event); 
			if(window.socket_retries > 0 && !window.hasOwnProperty("socket")){
				window.socket_retries --;
				socket_init(sockname);
			}else if(!window.hasOwnProperty("socket")){
				if(window.hasOwnProperty("socket_list")){
					//alert("SERVER DEAD");
					window.location.replace("http://192.168.177.77/roller/Improvement/inspection_v3.html");
				}
				delete(window.socket_list);
			}
		}
		window.socket_list[sockname].onopen = async (event) => {
			if(window.hasOwnProperty("socket")){
				if(!window.socket.readyState){
					console.log("CONNECTED");
					window.socket_retries = 10;
					window.socket = window.socket_list[sockname];
				}else{
					console.log("ABANDON", sockname);
				}
			}else{
				console.log("CONNECTED", sockname);
				window.socket_retries = 10;
				window.socket = window.socket_list[sockname];
			}
		}
	}
}

window.in_progress_fsdn = new Object;
window.fsdn_result = new Object;
//-- fsdn(event, func, func1, func2) --
// event = input type=file element
// func = function to execute on start
// func1 = function to execute after image resized
// func2 = function to execute after socket complete
function fsdn(event, func, func1, func2){
	let file = event.files[0];
//	if(!window.hasOwnProperty("socket")){
//		return new Promise((resolve, reject) => {resolve(new Object)});
//	}
//	if(!file || !window.socket.readyState == 1){
//		return new Promise((resolve, reject) => {resolve(new Object)});
//	}
	if(window.in_progress_fsdn.hasOwnProperty(event.id)){
		console.log("Not finished");
		return new Promise((resolve, reject) => {resolve(new Object)});
	}
	window.in_progress_fsdn[event.id] = new Date;
	if(func){
		console.log("exec FUNC");
		func();
	}
	window.fsdn_result[event.id] = new Promise((resolve) => {
		console.log("start FSD");
		let fsd = filesize_down(event, window.FSDN_size);
		event.setAttribute("disabled", "true");
		fsd.then(function(value){
			console.log("done FSD");
			if(!value){
				event.removeAttribute("disabled");
				resolve(new Object);
			}else{
				if(func1){
					func1(value);
				}
				data = new Object;
				let args = new Object;
				args["input_file_id"] = event.id;
				args["anchor_size"] = window.ANCHOR_SZ;
				args["nms_iou"] = window.NMS_IOU;
				args["image_size"] = window.IMAGE_SZ;
				args["image_format"] = window.IMAGE_FORMAT;
				args["segment_minimum_ratio"] = window.SEGMENT_MIN_RATIO;
				data["rawdata_only"] = true;
				data["filename"] = file.name;
				data["size"] = file.size;
				data["type"] = file.type;
				data["predict"] = true;
				data["tensorflow"] = true;
				data["args"] = args;
				data["base64"] = value;
				data["bytes"] = value;
				console.log("start SEND");
				if(!window.hasOwnProperty("socket")){
					console.log("FAILED TO ESTABLISH SOCKET CONNECTION");
					return new Promise((resolve, reject) => {resolve(new Object)});
				}
				if(!file || !window.socket.readyState == 1){
					console.log("FAILED ON FILE OR SOCKET FUNCTION");
					return new Promise((resolve, reject) => {resolve(new Object)});
				}
				send(data, function(v){
					console.log("finish SEND");
					if(func2){
						func2(v);
					}
					let img1 = new Image();
					img1.onload = function () {
						const naturalWidth = img1.width;
						const naturalHeight = img1.height;
						//console.log(naturalWidth, naturalHeight);
						let IMAGE = document.createElement("canvas");
						IMAGE.width = naturalWidth;
						IMAGE.height = naturalHeight;
						let im_ctx = IMAGE.getContext("2d");
						im_ctx.drawImage(img1, 0, 0, naturalWidth, naturalHeight);
						let tags = new Array;
						if(v.hasOwnProperty("rawdata")){
							for(var i = 0; i < v.rawdata.length; i ++){
								let obj = v.rawdata[i]; 
								if(window.ALLOWABLE_TAGS){
									//console.log(">>>", window.ALLOWABLE_TAGS.indexOf(obj.tag));
									if(window.ALLOWABLE_TAGS.indexOf(obj.tag)== -1){
										continue;
									}
								}
								let x1 = (obj.x1 / obj.w) * naturalWidth;
								let y1 = (obj.y1 / obj.h) * naturalHeight;
								let xw = ((obj.x2 - obj.x1) / obj.w) * naturalWidth;
								let yh = ((obj.y2 - obj.y1) / obj.h) * naturalHeight
								im_ctx.strokeStyle = obj.color;
								im_ctx.fillStyle = obj.color;
								im_ctx.lineWidth = 3;
								im_ctx.strokeRect(x1, y1, xw, yh);
								im_ctx.lineWidth = 2;
								im_ctx.font = "21px Comic Sans MS";
								im_ctx.fillText(obj.tag, x1, y1 -5);
								tags.push(obj.tag);
							}
						}
						let result = new Object;
						result.tag_result = tags;
						result.socket_rawdata = v;
						result.image_dataUrl = IMAGE.toDataURL();
						result.rawimage_dataUrl = value;
						window.setTimeout(function(){
							event.removeAttribute("disabled");
						}, 1000);
						//event.value = null;
						console.log("RESOLVING");
						resolve(result);
					};	
					img1.src = value;
				});
			}
		});
	});
	window.fsdn_result[event.id].then(e =>{
		if(e){
			delete(window.in_progress_fsdn[event.id]);
		}
	});
	return window.fsdn_result[event.id]
}

const send = (data, func) =>{
	if(socket.readyState != 1){
		socket_init();
		timer = setTimeout(function(){
			send(data, func);
		}, 1000);
		return false;
	}
	if(window.socket_query && socket.readyState){
		timer = setTimeout(function(){
			send(data, func);
		}, 1000);
		return false;
	}
	return new Promise((resolve) => {
		window.socket.send(JSON.stringify(data));
		window.socket_query = true;
		window.socket.onmessage = async (event) => {
			result = JSON.parse(event.data);
			resolve(func(result));
			window.socket_query = false;
		}
	});
}

function load_weight(){
	data = new Object;
	args = new Object;
	args["server"] = "load_weight";
	data["args"] = args;
	window.socket.send(JSON.stringify(data));
}












//https://straussengineering.ch/posts/javascript-bilinear-image-interpolation/
function ivect(ix, iy, w) {
	// byte array, r,g,b,a
	return((ix + w * iy) * 4);
}

function bilinear(srcImg, destImg, scale) {
	// c.f.: wikipedia english article on bilinear interpolation
	// taking the unit square, the inner loop looks like this
	// note: there's a function call inside the double loop to this one
	// maybe a performance killer, optimize this whole code as you need
	function inner(f00, f10, f01, f11, x, y) {
		var un_x = 1.0 - x; 
		var un_y = 1.0 - y;
		return (f00 * un_x * un_y + f10 * x * un_y + f01 * un_x * y + f11 * x * y);
	}
	var i, j;
	var iyv, iy0, iy1, ixv, ix0, ix1;
	var idxD, idxS00, idxS10, idxS01, idxS11;
	var dx, dy;
	var r, g, b, a;
	for (i = 0; i < destImg.height; ++i) {
		iyv = i / scale;
		iy0 = Math.floor(iyv);
		// Math.ceil can go over bounds
		iy1 = ( Math.ceil(iyv) > (srcImg.height-1) ? (srcImg.height-1) : Math.ceil(iyv) );
		for (j = 0; j < destImg.width; ++j) {
			ixv = j / scale;
			ix0 = Math.floor(ixv);
			// Math.ceil can go over bounds
			ix1 = ( Math.ceil(ixv) > (srcImg.width-1) ? (srcImg.width-1) : Math.ceil(ixv) );
			idxD = ivect(j, i, destImg.width);
			// matrix to vector indices
			idxS00 = ivect(ix0, iy0, srcImg.width);
			idxS10 = ivect(ix1, iy0, srcImg.width);
			idxS01 = ivect(ix0, iy1, srcImg.width);
			idxS11 = ivect(ix1, iy1, srcImg.width);
			// overall coordinates to unit square
			dx = ixv - ix0; dy = iyv - iy0;
			// I let the r, g, b, a on purpose for debugging
			r = inner(srcImg.data[idxS00], srcImg.data[idxS10],
				srcImg.data[idxS01], srcImg.data[idxS11], dx, dy);
			destImg.data[idxD] = r;
			g = inner(srcImg.data[idxS00+1], srcImg.data[idxS10+1],
				srcImg.data[idxS01+1], srcImg.data[idxS11+1], dx, dy);
			destImg.data[idxD+1] = g;
			b = inner(srcImg.data[idxS00+2], srcImg.data[idxS10+2],
				srcImg.data[idxS01+2], srcImg.data[idxS11+2], dx, dy);
			destImg.data[idxD+2] = b;
			a = inner(srcImg.data[idxS00+3], srcImg.data[idxS10+3],
				srcImg.data[idxS01+3], srcImg.data[idxS11+3], dx, dy);
			destImg.data[idxD+3] = a;
		}
	} 
}

var BicubicInterpolation = (function(){
    return function(x, y, values){
        var i0, i1, i2, i3;
        i0 = TERP(x, values[0][0], values[1][0], values[2][0], values[3][0]);
        i1 = TERP(x, values[0][1], values[1][1], values[2][1], values[3][1]);
        i2 = TERP(x, values[0][2], values[1][2], values[2][2], values[3][2]);
        i3 = TERP(x, values[0][3], values[1][3], values[2][3], values[3][3]);
        return TERP(y, i0, i1, i2, i3);
    };
    /* Yay, hoisting! */
    function TERP(t, a, b, c, d){
		return 0.5 * (c - a + (2.0*a - 5.0*b + 4.0*c - d + (3.0*(b - c) + d - a)*t)*t)*t + b;
		//return  desmosBicubic(0.5, a, b, c, d);
    }
})();


function bicubic(srcImg, destImg, scale) {
    var i, j;
    var dx, dy;
    var repeatX, repeatY;
    var offset_row0, offset_row1, offset_row2, offset_row3;
    var offset_col0, offset_col1, offset_col2, offset_col3;
    var red_pixels, green_pixels, blue_pixels, alpha_pixels;
    for (i = 0; i < destImg.height; ++i) {
        iyv = i / scale;
        iy0 = Math.floor(iyv);
        // We have to special-case the pixels along the border and repeat their values if neccessary
        repeatY = 0;
        if(iy0 < 1) repeatY = -1;
        else if(iy0 > srcImg.height - 3) repeatY = iy0 - (srcImg.height - 3);
        for (j = 0; j < destImg.width; ++j) {
            ixv = j / scale;
            ix0 = Math.floor(ixv);
            // We have to special-case the pixels along the border and repeat their values if neccessary
            repeatX = 0;
            if(ix0 < 1) repeatX = -1;
            else if(ix0 > srcImg.width - 3) repeatX = ix0 - (srcImg.width - 3);
            offset_row1 = ((iy0)   * srcImg.width + ix0) * 4;
            offset_row0 = repeatY < 0 ? offset_row1 : ((iy0-1) * srcImg.width + ix0) * 4;
            offset_row2 = repeatY > 1 ? offset_row1 : ((iy0+1) * srcImg.width + ix0) * 4;
            offset_row3 = repeatY > 0 ? offset_row2 : ((iy0+2) * srcImg.width + ix0) * 4;
            offset_col1 = 0;
            offset_col0 = repeatX < 0 ? offset_col1 : -4;
            offset_col2 = repeatX > 1 ? offset_col1 : 4;
            offset_col3 = repeatX > 0 ? offset_col2 : 8;
            //Each offset is for the start of a row's red pixels
            red_pixels = [[srcImg.data[offset_row0+offset_col0], srcImg.data[offset_row1+offset_col0], srcImg.data[offset_row2+offset_col0], srcImg.data[offset_row3+offset_col0]],
                              [srcImg.data[offset_row0+offset_col1], srcImg.data[offset_row1+offset_col1], srcImg.data[offset_row2+offset_col1], srcImg.data[offset_row3+offset_col1]],
                              [srcImg.data[offset_row0+offset_col2], srcImg.data[offset_row1+offset_col2], srcImg.data[offset_row2+offset_col2], srcImg.data[offset_row3+offset_col2]],
                              [srcImg.data[offset_row0+offset_col3], srcImg.data[offset_row1+offset_col3], srcImg.data[offset_row2+offset_col3], srcImg.data[offset_row3+offset_col3]]];
            offset_row0++;
            offset_row1++;
            offset_row2++;
            offset_row3++;
            //Each offset is for the start of a row's green pixels
            green_pixels = [[srcImg.data[offset_row0+offset_col0], srcImg.data[offset_row1+offset_col0], srcImg.data[offset_row2+offset_col0], srcImg.data[offset_row3+offset_col0]],
                              [srcImg.data[offset_row0+offset_col1], srcImg.data[offset_row1+offset_col1], srcImg.data[offset_row2+offset_col1], srcImg.data[offset_row3+offset_col1]],
                              [srcImg.data[offset_row0+offset_col2], srcImg.data[offset_row1+offset_col2], srcImg.data[offset_row2+offset_col2], srcImg.data[offset_row3+offset_col2]],
                              [srcImg.data[offset_row0+offset_col3], srcImg.data[offset_row1+offset_col3], srcImg.data[offset_row2+offset_col3], srcImg.data[offset_row3+offset_col3]]];
            offset_row0++;
            offset_row1++;
            offset_row2++;
            offset_row3++;
            //Each offset is for the start of a row's blue pixels
            blue_pixels = [[srcImg.data[offset_row0+offset_col0], srcImg.data[offset_row1+offset_col0], srcImg.data[offset_row2+offset_col0], srcImg.data[offset_row3+offset_col0]],
                              [srcImg.data[offset_row0+offset_col1], srcImg.data[offset_row1+offset_col1], srcImg.data[offset_row2+offset_col1], srcImg.data[offset_row3+offset_col1]],
                              [srcImg.data[offset_row0+offset_col2], srcImg.data[offset_row1+offset_col2], srcImg.data[offset_row2+offset_col2], srcImg.data[offset_row3+offset_col2]],
                              [srcImg.data[offset_row0+offset_col3], srcImg.data[offset_row1+offset_col3], srcImg.data[offset_row2+offset_col3], srcImg.data[offset_row3+offset_col3]]];
            offset_row0++;
            offset_row1++;
            offset_row2++;
            offset_row3++;
            //Each offset is for the start of a row's alpha pixels
            alpha_pixels =[[srcImg.data[offset_row0+offset_col0], srcImg.data[offset_row1+offset_col0], srcImg.data[offset_row2+offset_col0], srcImg.data[offset_row3+offset_col0]],
                              [srcImg.data[offset_row0+offset_col1], srcImg.data[offset_row1+offset_col1], srcImg.data[offset_row2+offset_col1], srcImg.data[offset_row3+offset_col1]],
                              [srcImg.data[offset_row0+offset_col2], srcImg.data[offset_row1+offset_col2], srcImg.data[offset_row2+offset_col2], srcImg.data[offset_row3+offset_col2]],
                              [srcImg.data[offset_row0+offset_col3], srcImg.data[offset_row1+offset_col3], srcImg.data[offset_row2+offset_col3], srcImg.data[offset_row3+offset_col3]]];
            // overall coordinates to unit square
            dx = ixv - ix0; 
			dy = iyv - iy0;
            idxD = ivect(j, i, destImg.width);
            destImg.data[idxD] = BicubicInterpolation(dx, dy, red_pixels);
            destImg.data[idxD+1] =  BicubicInterpolation(dx, dy, green_pixels);
            destImg.data[idxD+2] = BicubicInterpolation(dx, dy, blue_pixels);
            destImg.data[idxD+3] = BicubicInterpolation(dx, dy, alpha_pixels);
        }
    }
}

function desmosBicubic(x, a, b, c, d){
	return b + (x * (c - (2 * a + 3 * b + d)/ 6)) + (Math.pow(x, 2) * (a + c - 2 * b)/ 2) + (Math.pow(x, 3) * ((b - c) / 2 + (d - a) / 6))
}

window.RESIZE_MODE = 0;
function fsdf(event, max_size){
	console.log("start FSDF");
	return new Promise((resolve) => {
		const input = event;
		let maxsize = 1000;
		if(max_size){
			maxsize = max_size
		}
		let file = event.files[0];
		if(!event.files[0]){
			resolve(false);
		}
		let original_file_size = event.files[0].size;
		let read = new FileReader();
		read.onloadend = function(){
			console.log("FSDF read.onloadend");
			if(read.result.substr(0, 10) != "data:image"){
				resolve(null);
			}
			let image = new Image();
			image.onload = function() {
				console.log("FSDF image.onload");
				let ratio = image.width / image.height;
				if(ratio > 1){
					if(image.width > maxsize){
						destWidth = maxsize;
						destHeight = parseInt(maxsize / ratio);
					}else{
						destWidth = image.width;
						destHeight = image.height;
					}
				}else{
					if(image.height > maxsize){
						destWidth = parseInt(maxsize * ratio);
						destHeight = maxsize;
					}else{
						destWidth = image.width;
						destHeight = image.height;
					}
				}
				let read_canvas, read_ctx, srcImg;
				if(window.RESIZE_MODE != 0){
					read_canvas = document.createElement("canvas");
					read_ctx = read_canvas.getContext("2d");
					read_canvas.width = image.width;
					read_canvas.height = image.height;
					read_ctx.drawImage(image, 0, 0, image.width, image.height);
					srcImg = read_ctx.getImageData(0, 0, image.width, image.height);
				}
				
				let canvas = document.createElement("canvas");
				let ctx = canvas.getContext("2d");
				canvas.width = destWidth;
				canvas.height = destHeight;
				let destImg = ctx.getImageData(0, 0, canvas.width, canvas.height);
				window.destImg = destImg;
				let scale = canvas.width / image.width;
				
				if(window.RESIZE_MODE == 1){
					bilinear(srcImg, destImg, scale);
					ctx.putImageData(destImg, 0, 0);
				}else if(window.RESIZE_MODE == 2){
					bicubic(srcImg, destImg, scale);
					ctx.putImageData(destImg, 0, 0);
				}else{
					ctx.imageSmoothingEnabled = true;
					ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
				}					
				
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
				var dt = null;
				var image_dataurl = null;
				if(original_file_size > newImageFileFromCanvas.size){
					image_dataurl = dataUrl;
				}else{
					image_dataurl = image.src;
				}
				try{
					if(original_file_size > newImageFileFromCanvas.size){
						dt = new DataTransfer();
						dt.items.add(newImageFileFromCanvas);
						input.files = dt.files;
					}else{
					}
				}catch(e){
				}
				resolve(image_dataurl);
			}
			image.src = read.result;
		}	
		read.readAsDataURL(file);
	})
}

async function filesize_down(event, max_size){
	const result = await fsdf(event, max_size);
	return result;
}