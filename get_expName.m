function [expName] = get_expName(encType,vocSz,varargin)
order=2;
switch encType
	case 'fisher'
		poolType = 'hor0';
		if isempty(varargin)
			expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d',encType,vocSz,poolType,order);
		else
			expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d',encType,vocSz,poolType,order);
			%expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_ignore_tp%.02f',encType,vocSz,poolType,order,varargin{1});
			%expName = sprintf('LFK_root_imsz480_enc%s_debug_voc%d_pool%s_he025_step3_o%drun%d',encType,vocSz,poolType,order,varargin{1});
		end
	case 'vq'
		if vocSz<=256
			poolType='spm';
		else
			poolType='hor';
		end
		expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_ignore_tp1.00',encType,vocSz,poolType,order);
	case 'gabor'
		assert(~isempty(varargin),'3 inputs required for gabor');
		sfMin = vocSz;
		sfMax = varargin{1};
		expName = sprintf('gabor_sfmn%d_sfmx%d',sfMin,sfMax);
		clear vocSz;	
end
end

